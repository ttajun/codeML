
import os
import json
import pickle
import textwrap
import hashlib

from datetime import datetime as dt, timedelta

from common import logger
log = logger.make_logger(__name__)


def get_program_path():
    return os.path.dirname(os.path.dirname(__file__))


def get_program_name():
    _, program_name = os.path.split(get_program_path())
    return program_name


def load_dictionary(path):
    with open(path, encoding='utf-8-sig') as f:
        return json.load(f)


def save_dictionary(obj, path):
    with open(path, 'w', encoding='utf-8-sig') as f:
        f.write(json.dumps(obj, ensure_ascii=False))


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def date_to_str(date):
    return str(date).replace("-", "")[:8]


def today_str():
    return dt.today().strftime('%Y%m%d')


def create_directory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        log.error('Creating directory. ' +  directory)


def remove_newline(text:str):
    try:
        ret = ' '.join(text.splitlines())
    except:
        ret = ''
    return ret


def remove_comma(text:str):
    try: 
        ret = text.replace(',', ' ')
    except:
        ret = ''
    return ret


def short_str(txt, len=50):
    if not txt: return ''
    return textwrap.shorten(txt, width=len, placeholder='')


def get_all_attr(driver, element):
    attrs = driver.execute_script(
        'var items = {}; for (index = 0; index < arguments[0].attributes.length; ++index) { items[arguments[0].attributes[index].name] = arguments[0].attributes[index].value }; return items;', 
        element
    )
    return attrs


def hash_uid(url):
    tmp_uid = f'{url}'.encode('utf-8-sig')
    ret = hashlib.md5(tmp_uid).hexdigest()
    return ret

