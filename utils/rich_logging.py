import logging
from rich.logging import RichHandler
from rich.console import Console


# Custom handler to format log messages with colors
class MyRichLogHandler(logging.Handler):
    LEVEL_MAPPING = {
        logging.DEBUG: "[blue]DEBUG[/blue]",
        logging.INFO: "[green]INFO[/green]",
        logging.WARNING: "[yellow]WARNING[/yellow]",
        logging.ERROR: "[red]ERROR[/red]",
        logging.CRITICAL: "[bold red]CRITICAL[/bold red]",
    }

    def __init__(self):
        super().__init__()
        self.console = Console(soft_wrap=True)
        self.formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    def emit(self, record):
        msg = self.format(record)
        self.console.print(msg)

    def format(self, record):
        levelname = self.LEVEL_MAPPING.get(record.levelno, str(record.levelno))
        record.levelname = levelname
        return super().format(record)


def get_logger(level=logging.ERROR):
    # Configure logging
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[MyRichLogHandler()],
    )

    logger = logging.getLogger("rich")
    # logger.setLevel(level)
    return logger


if __name__ == "__main__":
    log = get_logger()
    # Log messages at different levels
    log.debug("This is a debug message")
    log.info("This is an info message")
    log.warning("This is a warning message")
    log.error("This is an error message")
    log.critical("This is a critical message")
