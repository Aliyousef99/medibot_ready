import importlib
m = importlib.import_module('backend.app')
print('Imported backend.app; routes:', len(getattr(m,'app').routes))
print('Has inline_extract:', any(getattr(r,'path',None)=='/api/labs/inline_extract' for r in m.app.routes))
