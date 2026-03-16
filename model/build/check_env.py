import importlib
packages = ['scipy','PIL','tensorflow','numpy']
for p in packages:
    name = 'PIL' if p == 'PIL' else p
    try:
        m = importlib.import_module(name)
        v = getattr(m, '__version__', None)
        print(f"{p}: ok, version={v}")
    except Exception as e:
        print(f"{p}: error, {e}")
