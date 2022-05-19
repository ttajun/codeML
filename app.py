
import os, sys
sys.path.append(os.path.dirname(__file__))

# tensorflow warning 제거. tensorflow import 전에 실행해야 함 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

from common import const, logger
from ml import build_model, load, explore, vectorize
log = logger.make_logger(__name__)


def main():
    con = const.Const
    log.info(f'# codeML     : ver {con.VERSION}')
    log.info(f'# tensorflow : ver {tf.__version__}')

    # Step 1. 수집 - Gather Data

    # Step 2. 탐색 - Explore Data
    
    # Step 3. 준비 - Prepare Data (tokenize, vectorize)
    
    # Step 4. 모델 - Build, Train, and Evaluate Model
    
    # Step 5. 조정 - Tune Hyperparameters
    
    # Step 6. 배포 - Deploy Model

    return


if __name__ == '__main__':
    main()
