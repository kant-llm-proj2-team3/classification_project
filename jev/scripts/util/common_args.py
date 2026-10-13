import argparse

def common_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--advanced", action="store_true", help="심화 데이터셋 사용")
    p.add_argument("--topic", choices=["documents", "inquiries"], required=True)
    p.add_argument("--extended", action="store_true")
    return p