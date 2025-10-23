import pathlib
path = pathlib.Path("orchestration/prefect_flows.py")
text = path.read_text()
old = "    logger.info(\"Running %s\", \" ".join(cmd))\n    _run_command(cmd)"
new = "    logger.info(\"Running %s\", \" ".join(cmd))\n    logger.info(\"Symbols arg: %s\", symbols)\n    _run_command(cmd)"
path.write_text(text.replace(old, new))
