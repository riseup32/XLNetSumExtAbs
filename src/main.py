import easydict
import os
import time
import re
import warnings
from selenium import webdriver
from others.logging import init_logger
from train_abstractive import validate_abs, train_abs, baseline, test_abs, test_text_abs
from train_extractive import train_ext, validate_ext, test_ext, test_text_ext


url = "https://edition.cnn.com/"

driver = webdriver.Chrome(executable_path="C:/Users/user/chromedriver_win32/chromedriver")
driver.get(url)
time.sleep(3)


xpath = '''//*[@id="header-nav-container"]/div/div[1]/div/button'''
driver.find_element_by_xpath(xpath).click()
time.sleep(1)

# query = input("Search CNN :")
element = driver.find_element_by_id("header-search-bar")
element.send_keys(input("Search in CNN: "))

xpath = '''//*[@id="header-nav-container"]/div/div[2]/div/div[1]/form/button'''
driver.find_element_by_xpath(xpath).click()
time.sleep(3)


headline_list = driver.find_element_by_css_selector('div.cnn-search__results-list').find_elements_by_tag_name("a")
for i in range(10):
    print(i+1, ':', headline_list[2*i+1].text)

print()
idx = int(input("Choose an article: "))

while(True):
    xpath = '''/html/body/div[5]/div[2]/div/div[2]/div[2]/div/div[3]/div[%d]/div[2]/h3/a''' %(idx)
    driver.find_element_by_xpath(xpath).click()
    time.sleep(7)

    try:
        article = driver.find_element_by_id('body-text').text
        article = re.sub('\n', ' ', article)
        break
    except:
        pass



def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


task_num = int(input("Choose a task\n1. Extractive Summarization\n2. Abstractive Summarization\n\nChoose a summary method:"))
if(task_num == 1):
    task = 'ext'
elif(task_num == 2):
    task = 'abs'


if(task == 'ext'):
    article = article.replace('. ', '. <cls> <sep> ')
    fw = open('C:/Users/user/Documents/Text Summarizaion/PreSumm-master/raw_text/article.txt', 'w', encoding='utf-8')
    fw.write(article)
    fw.close()
elif(task == 'abs'):
    fw = open('C:/Users/user/Documents/Text Summarizaion/PreSumm-master/raw_text/article.txt', 'w', encoding='utf-8')
    fw.write(article)
    fw.close()


if(task == 'ext'):
    args = easydict.EasyDict({
        "task": 'ext',
        "encoder": 'xlnet',
        "mode": 'test_text',
        "xlnet_data_path": '../xlnet_data_new/cnndm',
        "model_path": '../models/',
        "result_path": '../logs/article_ext_sum',
        "temp_dir": '../temp',

        "batch_size": 3000,
        "test_batch_size": 200,

        "max_pos": 512,
        "use_interval": True,
        "large": False,
        "load_from_extractive": '',

        "sep_optim": True,
        "lr_xlnet": 0.002,
        "lr_dec": 0.2,
        "use_xlnet_emb": True,

        "share_emb": False,
        "finetune_xlnet": True,
        "dec_dropout": 0.2,
        "dec_layers": 6,
        "dec_hidden_size": 768,
        "dec_heads": 8,
        "dec_ff_size": 2048,

        "enc_hidden_size": 512,
        "enc_ff_size": 512,
        "enc_dropout": 0.2,
        "enc_layers": 6,

        "ext_dropout": 0.1,
        "ext_layers": 2,
        "ext_hidden_size": 768,
        "ext_heads": 8,
        "ext_ff_size": 2048,

        "label_smoothing": 0.1,
        "generator_shard_size": 32,
        "alpha": 0.95,
        "beam_size": 5,
        "min_length": 50,
        "max_length": 200,
        "max_tgt_len": 140,

        "param_init": 0,
        "param_init_glorot": True,
        "optim": 'adam',
        "lr": 2e-3,
        "beta1": 0.9,
        "beta2": 0.999,
        "warmup_steps": 10000,
        "warmup_steps_xlnet": 20000,
        "warmup_steps_dec": 10000,
        "max_grad_norm": 0,

        "save_checkpoint_steps": 1000,
        "accum_count": 5,
        "report_every": 50,
        "train_steps": 50000,
        "recall_eval": False,

        "visible_gpus": '0',
        "gpu_ranks": '0',
        "log_file": '../logs/test_xlnet_ext_cnndm',
        "seed": 777,

        "test_all": True,
        "test_from": 'C:/Users/user/Documents/Text Summarizaion/PreSumm-master/models/XLNetExt/model_step_47000.pt',
        "test_start_from": -1,

        "train_from": '',
        "reprot_rouge": True,
        "block_trigram": True
    })
elif(task == 'abs'):
    args = easydict.EasyDict({
        "task": 'abs',
        "encoder": 'xlnet',
        "mode": 'test_text',
        "xlnet_data_path": '../xlnet_data_new/cnndm',
        "model_path": '../models/',
        "result_path": '../logs/article_abs_sum',
        "temp_dir": '../temp',

        "batch_size": 3000,
        "test_batch_size": 200,

        "max_pos": 512,
        "use_interval": True,
        "large": False,
        "load_from_extractive": '',

        "sep_optim": True,
        "lr_xlnet": 0.002,
        "lr_dec": 0.2,
        "use_xlnet_emb": True,

        "share_emb": False,
        "finetune_xlnet": True,
        "dec_dropout": 0.2,
        "dec_layers": 6,
        "dec_hidden_size": 768,
        "dec_heads": 8,
        "dec_ff_size": 2048,

        "enc_hidden_size": 512,
        "enc_ff_size": 512,
        "enc_dropout": 0.2,
        "enc_layers": 6,

        "ext_dropout": 0.1,
        "ext_layers": 2,
        "ext_hidden_size": 768,
        "ext_heads": 8,
        "ext_ff_size": 2048,

        "label_smoothing": 0.1,
        "generator_shard_size": 32,
        "alpha": 0.95,
        "beam_size": 5,
        "min_length": 50,
        "max_length": 200,
        "max_tgt_len": 140,

        "param_init": 0,
        "param_init_glorot": True,
        "optim": 'adam',
        "lr": 2e-3,
        "beta1": 0.9,
        "beta2": 0.999,
        "warmup_steps": 10000,
        "warmup_steps_xlnet": 20000,
        "warmup_steps_dec": 10000,
        "max_grad_norm": 0,

        "save_checkpoint_steps": 1000,
        "accum_count": 5,
        "report_every": 50,
        "train_steps": 50000,
        "recall_eval": False,

        "visible_gpus": '0',
        "gpu_ranks": '0',
        "log_file": '../logs/test_xlnet_abs_cnndm',
        "seed": 777,

        "test_all": True,
        "test_from": 'C:/Users/user/Documents/Text Summarizaion/PreSumm-master/models/XLNetExtAbs/model_step_292000.pt',
        "test_start_from": -1,

        "train_from": '',
        "reprot_rouge": True,
        "block_trigram": True
    })



args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
args.world_size = len(args.gpu_ranks)
os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

init_logger(args.log_file)
device = "cpu" if args.visible_gpus == '-1' else "cuda"
device_id = 0 if device == "cuda" else -1


args.text_src = 'C:/Users/user/Documents/Text Summarizaion/PreSumm-master/raw_text/article.txt'
args.text_tgt = ''


if (args.task == 'abs'):
    if (args.mode == 'train'):
        train_abs(args, device_id)
    elif (args.mode == 'validate'):
        validate_abs(args, device_id)
    elif (args.mode == 'lead'):
        baseline(args, cal_lead=True)
    elif (args.mode == 'oracle'):
        baseline(args, cal_oracle=True)
    if (args.mode == 'test'):
        cp = args.test_from
        try:
            step = int(cp.split('.')[-2].split('_')[-1])
        except:
            step = 0
        test_abs(args, device_id, cp, step)
    elif (args.mode == 'test_text'):
        test_text_abs(args)

elif (args.task == 'ext'):
    if (args.mode == 'train'):
        train_ext(args, device_id)
    elif (args.mode == 'validate'):
        validate_ext(args, device_id)
    if (args.mode == 'test'):
        cp = args.test_from
        try:
            step = int(cp.split('.')[-2].split('_')[-1])
        except:
            step = 0
        test_ext(args, device_id, cp, step)
    elif (args.mode == 'test_text'):
        test_text_ext(args)

print("-"*100)
print("The task is over")
