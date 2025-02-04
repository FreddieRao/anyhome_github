import os
from opt import get_default_parser

parser = get_default_parser()
args = parser.parse_args()
os.environ["OPENAI_API_KEY"] = args.api_key