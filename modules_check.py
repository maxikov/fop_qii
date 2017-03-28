import imp

modules_needed = ["prettytable", "numpy"]
modules_optional = ["matplotlib"]
modules_weird = ["pyspark"]

messages = {
        "needed": "Module {} is needed but not found",
        "optional": "Module {} is not found, but only "+\
                "needed for some functionality",
        "weird": "Module {} is not found, but this "+\
                "may be due to weird setup (i.e. the "+\
                "the system may work nonetheless)"
           }

def test_module(module_name, error_message):
    try:
        imp.find_module(module_name)
    except ImportError:
        print error_message.format(module_name)

for module in modules_needed:
    test_module(module, messages["needed"])
for module in modules_optional:
    test_module(module, messages["optional"])
for module in modules_weird:
    test_module(module, messages["weird"])
