from operator import itemgetter
import sys

def read_top_tables(filename, n):
    """Read in the n top performing results from a given file"""
    results = []
    with open(filename) as data:
        for line in data:
            results.append(line.split(','))

    with open(filename) as data:
        reader = csv.reader(data)
        for line in reader:
            results.append((float(line[3]), int(line[0]), line[-1]))
    results.sort(reverse=True, key=itemgetter(0))
    return results[:n]

if __name__ == "__main__":
    data_filename = sys.argv[1]
    results = read_top_tables(data_filename, 10)
    for result in results:
        print(result[0], result[-1])
