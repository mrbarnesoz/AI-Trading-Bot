from pathlib import Path
text = Path('tradingbotui/tasks.py').read_text()
old = """            record = {

                **base_info,

                "timestamp": raw.get("timestamp") or _timestamp(),

                "pnl": raw.get("pnl"),

                "confidence": raw.get("confidence"),

                "volatility": raw.get("volatility"),

                "position_size": raw.get("position_size"),

                "meta": raw,

            }

            records.append(record)
"""
new = """            record = {

                **base_info,

                "timestamp": raw.get("timestamp") or _timestamp(),

                "pnl": raw.get("pnl"),

                "confidence": raw.get("confidence"),

                "volatility": raw.get("volatility"),

                "position_size": raw.get("position_size"),

                "timeframe": raw.get("timeframe") or base_info.get("timeframe"),

                "meta": raw,

            }

            records.append(record)
"""
if old not in text:
    raise SystemExit('pattern not found for raw block')
text = text.replace(old, new, 1)
old_hist = """            records.append({**base_info, "timestamp": _timestamp(), "pnl": value, "confidence": None, "volatility": None, "position_size": None, "meta": {"index": idx}})
"""
new_hist = """            records.append({**base_info, "timestamp": _timestamp(), "pnl": value, "confidence": None, "volatility": None, "position_size": None, "timeframe": base_info.get("timeframe"), "meta": {"index": idx}})
"""
if old_hist not in text:
    raise SystemExit('pattern not found for history block')
text = text.replace(old_hist, new_hist, 1)
Path('tradingbotui/tasks.py').write_text(text)
