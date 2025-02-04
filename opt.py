import argparse

def get_default_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ### LLM CONFIG ###
    group = parser.add_argument_group("LLM")
    group.add_argument('--model', required=False, default="deepseek/deepseek-r1:free")
    group.add_argument('--api_key',required=False, default="your api key")
    group.add_argument('--base_url',required=False, default="https://openrouter.ai/api/v1")

    return parser