import sys

from classification import ClassificationDefault, ClassificationFashion
from fitting import FitExample
from regression import RegressionDefault
from save_load import SaveLoadExample


def register_module():
    """
    注册可以使用的module
    :returns: 返回dict，key是(module type, module name)，value是module对象
    """
    # 创建模块对象，在这里添加
    registered_modules = [ClassificationDefault(), ClassificationFashion(), RegressionDefault(), FitExample(),
                          SaveLoadExample()]

    # 返回处理过的列表
    return {(m.model_type(), m.model_name()): m for m in registered_modules}


if '__main__' == __name__:
    modules = register_module()

    if len(sys.argv) > 2:
        module_type = sys.argv[1]
        module_name = sys.argv[2]
    else:
        print("usage: {} [module type] [module name]".format(sys.argv[0]))
        print("available module:")
        for k in modules.keys():
            print("{:<15}{:>15}".format(k[0], k[1]))
        sys.exit(-1)

    module_key = (module_type, module_name)
    if module_key not in modules:
        print("unrecognized module key: {},{}".format(module_type, module_name))
        sys.exit(-1)
    else:
        modules[module_key].run()
