"""支持 `python -m src.engine` 直接调用原生人像精修引擎。"""
from .pipeline import _cli

if __name__ == '__main__':
    _cli()
