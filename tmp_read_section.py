from pathlib import Path
text = Path(r'C:\AI_trading_Bot\Readme.md').read_text(encoding='utf-8')
start = text.index('## Funding-Aware Meta Selector')
end = text.index('## Extending the Bot')
print(repr(text[start:end]))
