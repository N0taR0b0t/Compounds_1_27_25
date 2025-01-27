import csv
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

def sanitize_csv(input_file, output_file=None):
    """Advanced CSV repair with field mismatch recovery"""
    try:
        # Create output filename
        output_file = output_file or f"{Path(input_file).stem}_Prepared.csv"

        # First pass: detect CSV structure
        with open(input_file, 'r', encoding='ISO-8859-1') as f:
            sample = ''.join([f.readline() for _ in range(10)])
            f.seek(0)
            
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample)
            has_header = sniffer.has_header(sample)
            
            reader = csv.reader(f, dialect)
            header = next(reader) if has_header else []
            n_cols = len(header)

        # Second pass: process with recovery
        with open(input_file, 'r', encoding='ISO-8859-1') as f_in, \
             open(output_file, 'w', newline='', encoding='utf-8') as f_out:

            reader = csv.reader(f_in, dialect)
            writer = csv.writer(f_out, dialect=dialect)
            
            if has_header:
                writer.writerow(header)
                next(reader)
            else:
                writer.writerow([f'col_{i}' for i in range(n_cols)])

            f_in.seek(0)
            total_lines = sum(1 for _ in f_in) - (1 if has_header else 0)
            f_in.seek(0)
            if has_header:
                next(reader)

            error_log = []
            with tqdm(total=total_lines, desc="Repairing CSV") as pbar:
                for row_num, row in enumerate(reader, 1):
                    try:
                        if len(row) > n_cols:
                            merged_row = row[:n_cols-1] + [','.join(row[n_cols-1:])]
                            error_log.append(f"Line {row_num}: Merged {len(row)-n_cols} extra fields")
                            writer.writerow(merged_row)
                        elif len(row) < n_cols:
                            padded_row = row + [''] * (n_cols - len(row))
                            error_log.append(f"Line {row_num}: Added {n_cols - len(row)} missing fields")
                            writer.writerow(padded_row)
                        else:
                            writer.writerow(row)
                        pbar.update(1)
                    except Exception as e:
                        error_log.append(f"Line {row_num}: Failed processing - {str(e)}")
                        pbar.update(1)

        print(f"\nSuccessfully processed {total_lines} rows")
        print(f"Mismatch errors handled: {len(error_log)}")
        print(f"Output file: {output_file}")

        return output_file, None, None

    except Exception as e:
        print(f"Critical error: {str(e)}")
        return None, None, None

if __name__ == "__main__":
    input_file = 'Compounds_1_27_25.csv'
    sanitize_csv(input_file)