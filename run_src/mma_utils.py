# Licensed under the MIT license.

from enum import Enum, unique
import re
from typing import Dict, Tuple
import jieba
from sklearn.metrics import f1_score
import numpy as np
from rouge import Rouge

@unique
class Node_Type(Enum):
    USER_QUESTION = "USER_QUESTION"
    REPHRASED_USER_QUESTION = "REPHRASED_USER_QUESTION"
    BASE_ANSWER = "BASE_ANSWER"
    DOMAIN_ANSWER = "DOMAIN_ANSWER"
    CONTEXT = "CONTEXT"
    REPHRASED_CONTEXT = "REPHRASED_CONTEXT"
    OST_STEP = "OST_STEP"


class GeneratorError(Exception):
    def __init__(self, source, io_input, io_output_list) -> None:
        super().__init__()

        self.source = source
        self.io_input = io_input
        self.io_output_list = io_output_list


def reach_terminal_context(answer: str):
    assert answer is not None

    if "无需更改" in answer:
        #! remember that: when the original question is answerable, please start the subquestion with "Now we can answer the question: "
        return True

    return False

def compute_rouge(hyps, refs):
    if hyps == '' or hyps == []:
        if type(refs) == str:
            hyps = '无内容'
        else:
            hyps = ['无内容']
    if type(hyps) == str or len(hyps) == 1:
        hyps = ' '.join(jieba.cut(hyps)) 
        refs = ' '.join(jieba.cut(refs))
        return Rouge().get_scores(hyps, refs)
    elif len(hyps) > 1:
        hyps = [' '.join(jieba.cut(h)) for h in hyps]
        hyps = [h if h.strip() != "" else "无内容" for h in hyps]
        refs = [' '.join(jieba.cut(r)) for r in refs]
        return Rouge().get_scores(hyps, refs)
    else:
        raise ValueError("hyps and refs should be a list")
    
def compute_flzx(predictions,references):
    """
    Compute the ROUGE-L score between the prediction and the reference
    """

    rouge_scores = compute_rouge(predictions, references)
    rouge_ls = [score["rouge-l"]["f"] for score in rouge_scores]
    average_rouge_l = sum(rouge_ls) / len(rouge_ls)
    return average_rouge_l


def compute_acc(predictions,references):
    ref_list=[]
    pred_list=[]
    for pred,ref in zip(predictions,references):
        if "####" in ref:
            ref_answer = re.search(r"#### (-?\d+\.?\d*)", ref).group(1)
            ref_answer.replace(",", "")
            ref_answer = float(ref_answer)
        else:
            try:
                ref = ref.replace(",", "")
                ref_answer = re.search('-?\d+(?:\.\d+)?',ref)
                if ref_answer:
                    ref_answer = ref_answer.group(0)
                    ref_answer = re.sub(r'[^\d\-.]', '', ref_answer)
                    ref_answer = float(ref_answer)
                else:
                    return 0
            except:
                print(f"=============\n\n{ref_answer}\n\n============")
        if isinstance(pred, list):
            pred = pred[0]
        idx = pred.find("The final answer is")
        if idx != -1:
            pred = pred[idx:]
            pred = pred.replace(',', '')
            pred_answer = re.search('-?\d+(?:\.\d+)?', pred)
            pred_answer = pred_answer.group(0) if pred_answer else ''
            pred_answer = re.sub(r'[^\d\-.]', '', pred_answer)
            pred_answer = float(pred_answer) if pred_answer else ''
            pred_list.append(pred_answer)
        else:
            pred_list.append('')
        ref_list.append(ref_answer)

    return sum([ref_list[i]==pred_list[i] for i in range(len(ref_list))])/len(ref_list)


def compute_f1(predictions,references, dataset_name):
    task_labels_dict = {
        # First
        'LEGAL_AR': ["264", "133", "234"],
        "LEGAL_ER": ['LB10', 'LB11', 'LB12', 'LB13', 'LB14', 'LB15', 'LB16', 'LB17', 'LB18', 'LB19',
                     'LB20', 'LB1', 'LB2', 'LB3', 'LB4', 'LB5', 'LB6', 'LB7', 'LB8', 'LB9'],
        "LEGAL_NER": 'ner',
        "LEGAL_CR": ['刑事', '民事'],
        # Second
        "LEGAL_CFM": ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
        "LEGAL_SCM": ['B', 'C'],
        "LEGAL_CP": ['69', '50', '124'],
        "LEGAL_PTP": ['A', 'B', 'C'],
        "LEGAL_CTP": ['A', 'B', 'C'],
        "LEGAL_LQA": ['A', 'B', 'C', 'D'],
    }
    labels = task_labels_dict[dataset_name]
    pred_list = []  # prediction label
    answer_list = []  # target label

    missing = 0
    for pred,ref in zip(predictions,references):
        answer_list.append(ref.lower())
        miss = 1
        for answer1 in labels:
            if answer1.lower() in pred.lower():
                pred_list.append(answer1.lower())
                miss -= 1
                break
        if miss == 1:
            missing += 1
            pred_list.append("missing")
    y_true = np.array(answer_list)
    y_pred = np.array(pred_list)
    labels_lower = list(set(answer_list))

    return f1_score(y_true, y_pred, average='weighted', labels=labels_lower)

def compute_scm(predictions,references):
    pred_answer = []
    ref_answer = []
    if type(predictions) == str:
        predictions = [predictions]
        references = [references]
    for pred in predictions:
        idx = pred.rfind("最终答案")
        if idx != -1:
            pred = pred[idx:]
            letters = re.findall(r"[A-Za-z]", pred)
            if len(letters) == 1:
                pred_answer.append(letters[0])
            else:
                pred_answer.append("P")
        else:
            pred_answer.append("P")
    for ref in references:
        letters = re.findall(r"[A-Za-z]", ref)
        if len(letters) == 1:
            ref_answer.append(letters[0])
        else:
            ref_answer.append("O")
    return compute_f1(pred_answer,ref_answer,"LEGAL_SCM")

def extract_choice(predictions,references):
    pred_answer = []
    ref_answer = []
    if type(predictions) == str:
        predictions = [predictions]
        references = [references]
    for pred in predictions:
        idx = pred.rfind("正确答案")
        if idx != -1:
            pred = pred[idx:]
            letters = re.findall(r"[A-Za-z]", pred)
            if len(letters) == 1:
                pred_answer.append(letters[0])
            else:
                pred_answer.append("P")
        else:
            pred_answer.append("P")
    for ref in references:
        letters = re.findall(r"[A-Za-z]", ref)
        if len(letters) == 1:
            ref_answer.append(letters[0])
        else:
            ref_answer.append("O")
    return pred_answer,ref_answer
def compute_lqa(predictions,references):
    pred_answer, ref_answer = extract_choice(predictions,references)
    return compute_f1(pred_answer,ref_answer,"LEGAL_LQA")

def compute_choice(predictions,references):
    pred_answer, ref_answer = extract_choice(predictions,references)
    return sum([ref_answer[i]==pred_answer[i] for i in range(len(ref_answer))])/len(ref_answer)
def compute_cfm(predictions,references):
    pred_answer = []
    ref_answer = []
    if type(predictions) == str:
        predictions = [predictions]
        references = [references]
    for pred in predictions:
        idx = pred.rfind("正确答案")
        if idx != -1:
            pred = pred[idx:]
            letters = re.findall(r"[A-Za-z]", pred)
            if len(letters) == 1:
                pred_answer.append(letters[0])
            else:
                pred_answer.append("P")
        else:
            pred_answer.append("P")
    for ref in references:
        letters = re.findall(r"[A-Za-z]", ref)
        if len(letters) == 1:
            ref_answer.append(letters[0])
        else:
            ref_answer.append("O")
    return compute_f1(pred_answer,ref_answer,"LEGAL_CFM")

def extract_jejs(predictions,references):
    pred_list, ref_list = [], []

    if type(predictions) == str:
        predictions = [predictions]
        references = [references]

    for prediction, answer in zip(predictions,references):
        if "上文涉及到的犯罪金额:" in answer:
            answer = answer.replace("上文涉及到的犯罪金额:", "")
            answer = answer.replace("元。", "")
            answer = float(answer)
        else:
            answer_digits = re.findall(r"\d+\.?\d*", answer)
            if answer_digits!=[]:
                try:
                    answer_digits = answer_digits[-1]
                    if "千元" in answer:
                        answer = float(answer_digits) * 1000
                    elif "万元" in answer:
                        answer = float(answer_digits) * 10000
                    else:
                        answer = float(answer_digits)
                except:
                    answer = 0.01
            else:
                answer = 0.01

        if "最终金额" in prediction:
            idx = prediction.rfind("最终金额")
            prediction = prediction[idx:]
            prediction_digits = re.findall(r"\d+\.?\d*", prediction)
            if prediction_digits!=[]:
                try:
                    prediction_digits = prediction_digits[-1]
                    if "千元" in prediction:
                        prediction = float(prediction_digits) * 1000
                    elif "万元" in prediction:
                        prediction = float(prediction_digits) * 10000
                    else:
                        prediction = float(prediction_digits)
                except:
                    prediction = 0.00
            else:
                prediction = 0.00
        else:
            prediction = 0.00
        pred_list.append(prediction)
        ref_list.append(answer)
    return pred_list, ref_list

def compute_jejs(predictions,references):

    pred_list, ref_list = extract_jejs(predictions,references)
    return sum([ref_list[i]==pred_list[i] for i in range(len(ref_list))])/len(ref_list)


def extract_zmyc(predictions, references):
    pred_list, ref_list = [], []
    if type(predictions) == str:
        predictions = [predictions]
        references = [references]
    for prediction, answer in zip(predictions, references):
        answer = answer.replace("罪名", "").replace(":", "").replace("：", "").replace("罪", "").replace("。", "").strip()
        
        idx = prediction.rfind("罪名")
        if idx != -1:
            prediction = prediction[idx:]
        else:
            prediction = '无'
        if "。" in prediction:
            idx = prediction.find("。")
            prediction = prediction[:idx]
        prediction = prediction.replace("罪名", "").replace(":", "").replace("：", "").replace("罪", "").replace("。", "").strip()
        if ";" in answer:
            ref_list.append(answer.split(";"))
        else:
            ref_list.append(answer.split("；"))
        if ";" in prediction:
            pred_list.append(prediction.split(";"))
        else:
            pred_list.append(prediction.split("；"))
    
    return pred_list, ref_list

def compute_f1_two_sets(pred_set, gt_set):
    precision = len(pred_set.intersection(gt_set)) / len(pred_set) if len(pred_set) > 0 else 0
    recall = len(pred_set.intersection(gt_set)) / len(gt_set) if len(gt_set) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return f1          

def compute_zmyc(predictions,references):
    option_list = ["侮辱", "违法发放贷款", "失火", "票据诈骗", "帮助犯罪分子逃避处罚", "重大责任事故", "对非国家工作人员行贿","非法制造、销售非法制造的注册商标标识", "非法制造、买卖、运输、邮寄、储存枪支、弹药、爆炸物", "非法获取公民个人信息","扰乱无线电通讯管理秩序", "非法持有、私藏枪支、弹药", "拒不执行判决、裁定", "虚开发票", "巨额财产来源不明","组织、领导、参加黑社会性质组织", "非法获取国家秘密", "以危险方法危害公共安全", "非法持有毒品","聚众扰乱公共场所秩序、交通秩序", "包庇毒品犯罪分子", "滥伐林木", "伪造公司、企业、事业单位、人民团体印章","非法占用农用地", "走私废物", "串通投标", "非法采伐、毁坏国家重点保护植物", "冒充军人招摇撞骗", "玩忽职守","重婚", "招收公务员、学生徇私舞弊", "组织、领导传销活动", "非法猎捕、杀害珍贵、濒危野生动物", "侵犯著作权","非法种植毒品原植物", "伪造、变造、买卖武装部队公文、证件、印章", "倒卖文物", "伪造、变造居民身份证", "滥用职权","诽谤", "猥亵儿童", "非法转让、倒卖土地使用权", "挪用公款", "污染环境", "出售、购买、运输假币", "敲诈勒索","高利转贷", "故意伤害", "持有、使用假币", "单位受贿", "强奸", "引诱、容留、介绍卖淫", "虐待","生产、销售伪劣农药、兽药、化肥、种子", "妨害公务", "容留他人吸毒", "拐骗儿童", "强制猥亵、侮辱妇女","非法处置查封、扣押、冻结的财产", "骗取贷款、票据承兑、金融票证", "强迫他人吸毒", "非法拘禁","非法携带枪支、弹药、管制刀具、危险物品危及公共安全", "绑架", "聚众斗殴", "破坏计算机信息系统","制造、贩卖、传播淫秽物品", "虐待被监管人", "贷款诈骗", "赌博", "徇私舞弊不征、少征税款","盗窃、抢夺枪支、弹药、爆炸物、危险物质", "故意杀人", "介绍贿赂", "提供侵入、非法控制计算机信息系统程序、工具","编造、故意传播虚假恐怖信息", "妨害作证", "强迫卖淫", "走私、贩卖、运输、制造毒品", "伪证", "拐卖妇女、儿童","过失损坏武器装备、军事设施、军事通信", "破坏广播电视设施、公用电信设施", "洗钱", "职务侵占", "倒卖车票、船票","抢劫", "侵占", "掩饰、隐瞒犯罪所得、犯罪所得收益", "徇私舞弊不移交刑事案件", "引诱、教唆、欺骗他人吸毒", "遗弃","生产、销售伪劣产品", "放火", "非法采矿", "对单位行贿", "盗窃、抢夺枪支、弹药、爆炸物", "破坏易燃易爆设备","妨害信用卡管理", "制作、复制、出版、贩卖、传播淫秽物品牟利", "金融凭证诈骗", "私分国有资产","走私国家禁止进出口的货物、物品", "假冒注册商标", "危险物品肇事", "走私普通货物、物品", "经济犯", "虚报注册资本","盗掘古文化遗址、古墓葬", "传播淫秽物品", "窝藏、包庇", "拒不支付劳动报酬", "行贿", "开设赌场", "传授犯罪方法","协助组织卖淫", "保险诈骗", "破坏生产经营", "破坏交通设施", "打击报复证人", "非法侵入住宅", "非国家工作人员受贿","过失致人重伤", "伪造、变造金融票证", "窝藏、转移、隐瞒毒品、毒赃", "帮助毁灭、伪造证据", "走私珍贵动物、珍贵动物制品","生产、销售假药", "逃税", "挪用特定款物", "聚众扰乱社会秩序", "组织、强迫、引诱、容留、介绍卖淫", "合同诈骗","非法生产、销售间谍专用器材", "破坏交通工具", "传播性病", "强迫交易", "隐匿、故意销毁会计凭证、会计帐簿、财务会计报告","非法组织卖血", "强迫劳动", "破坏电力设备", "销售假冒注册商标的商品", "收买被拐卖的妇女、儿童", "诬告陷害", "脱逃","非法经营", "徇私枉法", "信用卡诈骗", "生产、销售不符合安全标准的食品", "非法行医", "伪造货币", "动植物检疫徇私舞弊","单位行贿", "破坏监管秩序", "盗窃", "盗伐林木", "重大劳动安全事故", "非法吸收公众存款","非法制造、出售非法制造的发票", "非法狩猎", "组织卖淫", "非法买卖、运输、携带、持有毒品原植物种子、幼苗", "挪用资金","诈骗", "伪造、变造、买卖国家机关公文、证件、印章", "持有伪造的发票", "贪污", "非法生产、买卖警用装备","投放危险物质", "伪造、倒卖伪造的有价票证", "集资诈骗", "抢夺", "生产、销售有毒、有害食品", "非法捕捞水产品","过失致人死亡", "非法买卖制毒物品", "虚开增值税专用发票、用于骗取出口退税、抵扣税款发票", "寻衅滋事", "危险驾驶","故意毁坏财物", "招摇撞骗", "盗窃、侮辱尸体", "走私武器、弹药","非法收购、运输、加工、出售国家重点保护植物、国家重点保护植物制品", "非法出售发票", "劫持船只、汽车","受贿", "聚众哄抢", "交通肇事"]
    
    pred_list, ref_list = extract_zmyc(predictions, references)
    score_list = []
    abstentions = 0

    for pred,ref in zip(pred_list, ref_list):
        pred_option_list =[]
        for option in option_list:
            if option in  pred:
                pred_option_list.append(option)

        gt_set = set(ref)
        pred_set = set(pred_option_list)

        precision = len(pred_set.intersection(gt_set)) / len(pred_set) if len(pred_set) > 0 else 0
        recall = len(pred_set.intersection(gt_set)) / len(gt_set) if len(gt_set) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        score_list.append(f1)
    f1_score_average = sum(score_list) / len(score_list)
    return f1_score_average

def extract_agi(predictions, references):
    pred_list, ref_list = [], []
    if type(predictions) == str:
        predictions = [predictions]
        references = [references]
    for prediction, answer in zip(predictions, references):
        answer = answer.replace("最终答案", "").replace(":", "").replace("：", "").replace("。", "").replace("为", "").strip()
        
        idx = prediction.rfind("最终答案")
        if idx != -1:
            prediction = prediction[idx:]
        else:
            prediction = '无'
        prediction = prediction.replace("最终答案", "").replace(":", "").replace("：", "").replace("。", "").replace("为", "").replace(".","").strip()
        
        pred_list.append(prediction)
        ref_list.append(answer)
    return pred_list, ref_list

def compute_agi(predictions,references):

    pred_list, ref_list = extract_agi(predictions, references)
    return sum([ref_list[i].lower()==pred_list[i].lower() for i in range(len(ref_list))])/len(ref_list)

def compute_nl2(predictions,references):

    pred_list, ref_list = extract_agi(predictions, references)
    return sum([ref_list[i]==pred_list[i] for i in range(len(ref_list))])/len(ref_list)

def compute_ftbs(predictions,references):
    """
    Compute the ROUGE-L score between the prediction and the reference
    """

    rouge_scores = compute_rouge(predictions, references)
    rouge_ls = [score["rouge-l"]["f"] for score in rouge_scores]
    average_rouge_l = sum(rouge_ls) / len(rouge_ls)
    return average_rouge_l