# Tabular processing with PAI-RAG

## PaiCSVReader

PaiCSVReader(concat_rows=True, row_joiner="\n", csv_config={})

### Parameters:

**concat_rows:** _bool, default=True._
Whether to concatenate rows into one document.

**row_joiner:** _str, default="\n"._
The separator used to join rows.

**header:** _None or int, list of int, default 0._
row (0-indexed) to use for the column labels of the parsed DataFrame. If a list of integers is passed those row
positions will be combined into a MultiIndex. Use None if there is no header.

### Functions:

load_data(file: Path, extra_info: Optional[Dict] = None, fs: Optional[AbstractFileSystem] = None)

## PaiPandasCSVReader

PaiPandasCSVReader(concat_rows=True, row_joiner="\n", pandas_config={})

### Parameters:

**concat_rows:** _bool, default=True._
Whether to concatenate rows into one document.

**row_joiner:** _str, default="\n"._
The separator used to join rows.

**pandas_config:** _dict, default={}._
The configuration of pandas.read_csv.
Refer to https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html for more information.
Set to empty dict by default, this means pandas will try to figure out the separators, table head, etc. on its own.

#### one important parameter:

**header:** _None or int, list of int, default 0._
Row (0-indexed) to use for the column labels of the parsed DataFrame. If a list of integers is passed those row
positions will be combined into a MultiIndex. Use None if there is no header.

### Functions:

load_data(file: Path, extra_info: Optional[Dict] = None, fs: Optional[AbstractFileSystem] = None)

## PaiPandasExcelReader

PaiPandasExcelReader(concat_rows=True, row_joiner="\n", pandas_config={})

### Parameters:

**concat_rows:** _bool, default=True._
Whether to concatenate rows into one document.

**row_joiner:** _str, default="\n"._
The separator used to join rows.

**pandas_config:** _dict, default={}._
The configuration of pandas.read_csv.
Refer to https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html for more information.
Set to empty dict by default, this means pandas will try to figure out the separators, table head, etc. on its own.

#### one important parameter:

**header:** _None or int, list of int, default 0._
Row (0-indexed) to use for the column labels of the parsed DataFrame. If a list of integers is passed those row
positions will be combined into a MultiIndex. Use None if there is no header.

### Functions:

load_data(file: Path, extra_info: Optional[Dict] = None, fs: Optional[AbstractFileSystem] = None)
only process the first sheet
