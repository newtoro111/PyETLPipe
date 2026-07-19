#
# Chain of responsibility to implement a three step pipeline
#
# Author: Ronn Newton
# Date Written: 2024-06-10
#
#
# Initial data created manually then augmented with ChatGPT generated data to get to ~1200 rows.
#
# The data is processed by each handler in the chain then written to a new file that is passed to the next handler.
#
# The gold handler writes two output files, one for the billing data and one for the payment data.
#
# Afterwards, the gold handler writes a parquet file for the billing data.
#
# Analysis is last and includes two charts, one for count of unique accounts by state and one for total billed amount by state.
#
# ChatGPT was used to generate the code for the logging and charting.
#
# Inspired by Chain Of Responsibility Design Pattern | Python Example at https://www.youtube.com/watch?v=QOW1IN8i8J8  and Refactoring a 500-Line Method with the Pipeline Pattern
# https://www.youtube.com/watch?v=RfknMfzTUbo&t=496s
#
# Updates:
#
# Author  - Date - Note:
# Ronn - 2024-06-10: Initial version.
# Ronn - 2024-06-26: Added logging and charting.
# Ronn - 2026-07-19: Ensure Account is int64 after coercion.


from __future__ import annotations
import datetime
import logging
import os
import time
from abc import ABC, abstractmethod
from functools import wraps

import pandas as pd
from pathlib import Path
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Some constants for logging and versioning.

LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LAST_UPDATED: str = "2026-07-19 21:34:00Z"


def configure_logging(
    level: int = logging.INFO,
    log_file: str | None = None,
) -> None:
    """
    Configure standard Python logging for the application.

    This should be called once near the application entry point.
    Use UTC instead of local time for all logging timestamps
    """    
    logging.Formatter.converter = time.gmtime
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


# Module-level logger.
logger = logging.getLogger(__name__)


def timer(func):
    """
    Log the elapsed execution time for a function or method.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        method_logger = logging.getLogger(f"{func.__module__}.{func.__qualname__}")
        start = time.perf_counter()

        method_logger.debug("Starting %s", func.__qualname__)

        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            method_logger.info(
                "%s completed in %.3f ms",
                func.__qualname__,
                elapsed * 1000,
            )

    return wrapper


class Handler(ABC):
    """
    Abstract handler that defines the interface for each pipeline stage.

    Logging is included here so all concrete handlers inherit a consistent
    logger and consistent chain handoff behavior.
    """

    def __init__(self, successor: Handler | None = None) -> None:
        self.successor = successor

        # Class-level logger, for example:
        # __main__.BronzeHandler, __main__.SilverHandler, etc.
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        if successor is None:
            self.logger.debug("Initialized without a successor")
        else:
            self.logger.debug(
                "Initialized with successor: %s",
                successor.__class__.__name__,
            )

    @abstractmethod
    def process_stage(self, source: str) -> None:
        """
        Process one stage of the pipeline.

        This abstract method has a body intentionally. Concrete classes can call
        super().process_stage(source) to log common stage-entry information.
        """
        self.logger.debug(
            "Entering %s.process_stage with source=%r",
            self.__class__.__name__,
            source,
        )

    def next_stage(self, source: str) -> None:
        """
        Pass control to the next handler in the chain, if one exists.
        """
        if self.successor is None:
            self.logger.info(
                "No successor configured after %s; chain processing is complete",
                self.__class__.__name__,
            )
            return

        self.logger.info(
            "Passing %r from %s to %s",
            source,
            self.__class__.__name__,
            self.successor.__class__.__name__,
        )
        self.successor.process_stage(source)


class BronzeHandler(Handler):
    @timer
    def process_stage(self, source: str) -> None:
        super().process_stage(source)

        self.logger.info("Bronze stage started with source=%r", source)

        if source == "bronze":
            self.logger.info("BronzeHandler: handling bronze source")
        elif source == "any":
            self.logger.info("BronzeHandler: handling any source")

        output_file = "bronze.csv"

        try:
            df = pd.read_csv(source, delimiter="|")
            self.logger.info("Loaded DataFrame from %s", source)
            self.logger.debug("Initial DataFrame shape: %s", df.shape)
            self.logger.debug("Initial DataFrame preview:\n%s", df.head())

            account_null_count = df["Account"].isnull().sum()
            self.logger.info(
                "Null values in Account column before cleanup: %s",
                account_null_count,
            )

            df = df[df["Account"].notnull()]
            self.logger.debug("DataFrame shape after Account null cleanup: %s", df.shape)

            total_null_count = df.isnull().sum().sum()
            self.logger.info("Total null values in %s: %s", source, total_null_count)

            # Drop rows with non-numeric values in Account and BilledAmount columns.
            df["Account"] = pd.to_numeric(df["Account"], errors="coerce")
            df.dropna(subset=["Account"], inplace=True)
            df["Account"] = df["Account"].astype("int64")  # Ronn - 2026-07-19: Ensure Account is int64 after coercion


            df["BilledAmount"] = pd.to_numeric(df["BilledAmount"], errors="coerce")
            df.dropna(subset=["BilledAmount"], inplace=True)

            self.logger.debug(
                "DataFrame shape after numeric cleanup: %s",
                df.shape,
            )

            df.to_csv(output_file, index=False)
            self.logger.info("Bronze output written to %s", output_file)

        except FileNotFoundError:
            self.logger.exception("File not found: %s", source)
            return
        except pd.errors.EmptyDataError:
            self.logger.exception("The file is empty: %s", source)
            return
        except pd.errors.ParserError:
            self.logger.exception("Could not parse input file: %s", source)
            return
        except KeyError:
            self.logger.exception(
                "Required column is missing while processing %s",
                source,
            )
            return
        except Exception:
            self.logger.exception(
                "Unexpected error in BronzeHandler while processing %s",
                source,
            )
            return

        self.next_stage(output_file)


class SilverHandler(Handler):
    @timer
    def process_stage(self, source: str) -> None:
        super().process_stage(source)

        self.logger.info("Silver stage started with source=%r", source)

        if source == "silver":
            self.logger.info("SilverHandler: handling silver source")
        elif source == "any":
            self.logger.info("SilverHandler: handling any source")

        output_file = "silver.csv"

        try:
            df = pd.read_csv(source)
            self.logger.info("Loaded DataFrame from %s", source)
            self.logger.debug("Initial DataFrame shape: %s", df.shape)
            self.logger.debug("Initial DataFrame preview:\n%s", df.head())

            amount = pd.to_numeric(df["BilledAmount"], errors="coerce")
            rows_before = len(df)

            df.drop(df[amount > 10_000_000].index, inplace=True)

            rows_removed = rows_before - len(df)
            self.logger.info(
                "Removed %s rows with BilledAmount greater than 10,000,000",
                rows_removed,
            )
            self.logger.debug("DataFrame shape after silver cleanup: %s", df.shape)

            df.to_csv(output_file, index=False)
            self.logger.info("Silver output written to %s", output_file)

        except FileNotFoundError:
            self.logger.exception("File not found: %s", source)
            return
        except pd.errors.EmptyDataError:
            self.logger.exception("The file is empty: %s", source)
            return
        except pd.errors.ParserError:
            self.logger.exception("Could not parse input file: %s", source)
            return
        except KeyError:
            self.logger.exception(
                "Required column is missing while processing %s",
                source,
            )
            return
        except Exception:
            self.logger.exception(
                "Unexpected error in SilverHandler while processing %s",
                source,
            )
            return

        self.next_stage(output_file)


class GoldHandler(Handler):
    @timer
    def process_stage(self, source: str) -> None:
        super().process_stage(source)

        self.logger.info("Gold stage started with source=%r", source)

        if source == "gold":
            self.logger.info("GoldHandler: handling gold source")
        elif source == "any":
            self.logger.info("GoldHandler: handling any source")

        gold_file = "gold.csv"
        payment_file = "goldpymt.csv"
        parquet_file = "gold.parquet"

        try:
            df = pd.read_csv(source)
            self.logger.info("Loaded DataFrame from %s", source)
            self.logger.debug("Initial DataFrame shape: %s", df.shape)
            self.logger.debug("Initial DataFrame preview:\n%s", df.head())

            # Split the incoming dataframe into two dataframes.
            payment_df = df[["Account", "DatePaid", "AmtPaid"]].copy()
            payment_df.to_csv(payment_file, index=False)
            self.logger.info("Payment output written to %s", payment_file)

            df.drop(
                columns=["DatePaid", "AmtPaid"],
                inplace=True,
                errors="ignore",
            )

            df.to_csv(gold_file, index=False)
            self.logger.info("Gold CSV output written to %s", gold_file)

            df.to_parquet(parquet_file, index=False)
            self.logger.info("Gold Parquet output written to %s", parquet_file)

        except FileNotFoundError:
            self.logger.exception("File not found: %s", source)
            return
        except pd.errors.EmptyDataError:
            self.logger.exception("The file is empty: %s", source)
            return
        except pd.errors.ParserError:
            self.logger.exception("Could not parse input file: %s", source)
            return
        except KeyError:
            self.logger.exception(
                "Required column is missing while processing %s",
                source,
            )
            return
        except Exception:
            self.logger.exception(
                "Unexpected error in GoldHandler while processing %s",
                source,
            )
            return

        self.next_stage(gold_file)


def build_chain() -> Handler:
    """
    Wire up the chain of responsibility.
    """
    gold_handler = GoldHandler(successor=None)
    silver_handler = SilverHandler(successor=gold_handler)
    bronze_handler = BronzeHandler(successor=silver_handler)

    logger.debug("Pipeline chain created: BronzeHandler -> SilverHandler -> GoldHandler")

    return bronze_handler


def plot_unique_accounts_by_state(df: pd.DataFrame) -> None:
    """
    Graph the number of unique accounts by state using a bar chart.
    """
    logger.info("Creating Unique Accounts by State chart")

    account_counts = (
        df.groupby("State")["Account"]
        .nunique()
        .sort_index()
    )

    plt.figure(figsize=(16, 6))
    account_counts.plot(kind="bar")
    as_of_date = datetime.date.today().strftime("%Y-%m-%d")

    plt.title(f"Total Unique Accounts by State as of {as_of_date}")
    plt.xlabel("State")
    plt.ylabel("Number of Accounts")

    plt.tight_layout()
    plt.savefig("AccountsByState.png", dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("Saved chart to AccountsByState.png")


def plot_billed_amount_by_state(df: pd.DataFrame) -> None:
    """
    Graph total billed amount by state using a bar chart.
    """
    logger.info("Creating Total Billed Amount by State chart")

    billed_by_state = (
        df.groupby("State")["BilledAmount"]
        .sum()
        .sort_index()
    )

    plt.figure(figsize=(16, 6))
    ax = billed_by_state.plot(kind="bar")

    as_of_date = pd.to_datetime(df["Date"]).max().strftime("%Y-%m-%d")  
    
    plt.title(f"Total Billed Amount by State as of {as_of_date}")
    plt.xlabel("State")
    plt.ylabel("Total Billed Amount ($)")

    plt.xticks(rotation=45)

    plt.gca().yaxis.set_major_formatter(
        FuncFormatter(lambda y, _: f'${y:,.0f}')
    )

    # Add horizontal grid lines at each y-axis label amount
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)

        # Add total billed amount labels above each state bar
    for bar in ax.patches:
        height = bar.get_height()

        ax.annotate(
            f"${height:,.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),              # 3 points above the bar
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90       # Rotate the label 90 degrees to be vertical
        )

    # Add extra space so labels do not get cut off
    ax.set_ylim(top=billed_by_state.max() * 1.15)

    plt.tight_layout()
    plt.savefig("BilledAmountByState.png", dpi=300, bbox_inches="tight")

    # Display the plot in interactive environments (e.g., Jupyter notebooks)
    plt.show()
    
    # Close all open figures to free up memory and avoid display issues in interactive environments
    plt.close('all')               

    logger.info("Saved chart to BilledAmountByState.png")


def read_gold_outputs(parquet_file: str = "gold.parquet") -> pd.DataFrame | None:
    """
    Read the Gold Parquet output and return a DataFrame.
    """
    if not os.path.exists(parquet_file):
        logger.warning(
            "Skipping Parquet read because %s does not exist",
            parquet_file,
        )
        return None

    logger.info("Reading %s with pyarrow", parquet_file)
    parquet_table = pq.read_table(parquet_file)
    logger.debug("PyArrow preview:\n%s", parquet_table.to_pandas().head())

    logger.info("Reading %s with pandas", parquet_file)
    df = pd.read_parquet(parquet_file)
    logger.debug("Pandas preview:\n%s", df.head())

    return df


def main(source_file: str = "source.csv") -> None:
    """
    Application entry point.
    """
    configure_logging(level=logging.INFO, log_file="pipeline.log")


    logger.info(f"Starting medallion pipeline with {Path(__file__).name} last updated {LAST_UPDATED}")

    cwd = os.getcwd()
    logger.info("Current working directory: %s", cwd)
    print(f"Current working directory: {cwd}")

    logger.info("Source file: %s", source_file)

    pipeline = build_chain()
    pipeline.process_stage(source_file)

    df = read_gold_outputs("gold.parquet")

    if df is None:
        logger.warning("No Gold DataFrame available; skipping charts")
        return

    plot_unique_accounts_by_state(df)
    plot_billed_amount_by_state(df)

    logger.info("Pipeline completed")


if __name__ == "__main__":
    main()
