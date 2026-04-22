__description__ = """
    Converte um arquivo csv no formato mais popular usado no Windows (ISO-8859-1, 
    separado por ponto e vírgula) para um arquivo csv no formato mais popular usado no Linux 
    (UTF-8, separado por vírgula).
"""

import sys
import csv
import argparse

import pandas as pd

def main(csv_path, *, sep=';', encoding='ISO-8859-1', to_file=False):
    df = pd.read_csv(csv_path, sep=sep, encoding=encoding)
    if to_file:
        df.to_csv(csv_path.replace('.csv', ' (utf-8).csv'), index=False, sep=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC, encoding='utf-8')
    else:
        df.to_csv(sys.stdout.buffer, index=False, sep=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC, encoding='utf-8')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__description__
    )

    parser.add_argument(
        '--csv-path', action='store', required=True,
        help='Caminho para um arquivo csv.'
    )

    parser.add_argument(
        '--sep', action='store', required=False, default=';',
        help='Separador usado no arquivo csv. O padrão é \';\'.'
    )

    parser.add_argument(
        '--encoding', action='store', required=False, default='ISO-8859-1',
        help='Codificação usada no arquivo csv. O padrão é \'ISO-8859-1\'.'
    )

    parser.add_argument(
        '--to-file', action='store_true', default=False,
        help='Opcional - se a conversão deve ser escrita para um arquivo. Se não, imprime na tela.'
    )

    args = parser.parse_args()
    main(args.csv_path, sep=args.sep, encoding=args.encoding, to_file=args.to_file)