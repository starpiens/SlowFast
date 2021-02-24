
def main():
    labels = []
    with open("test.csv", "r") as f:
        for line in f.read().splitlines()[1:-1]:
            label, _, _, _, _ = line.split(',')
            label += '\n'
            if label not in labels:
                labels.append(label)
    labels.sort()
    with open("labels.csv", "w") as f:
        f.writelines(labels)
    
if __name__ == '__main__':
    main()
