
import os, sys
sys.path.append(os.path.dirname(__file__))

# tensorflow warning 제거. tensorflow import 전에 실행해야 함 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from zUtil import registry, const, logger
log = logger.make_logger(__name__)


def main():
    con = const.Const
    reg = registry.registry
    flow_name = con.FLOW_MLCC_TEXT
    log.debug(f'flow_name: {flow_name}')

    # flow class 생성
    flo = reg[flow_name]()
    flo.flow()

    return


if __name__ == '__main__':
    main()
