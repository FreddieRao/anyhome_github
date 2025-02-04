import argparse

def get_default_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ### LLM CONFIG ###
    group = parser.add_argument_group("LLM")
    group.addargument('--model', required=False, default="deepseek-chat")
    group.addargument('--api_key',required=False, default="")
    group.addargument('--base_url',required=False, default="https://api.deepseek.com")