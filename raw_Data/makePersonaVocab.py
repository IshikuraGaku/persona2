import progressbar
import io
import re
import argparse


def main():
    print("OK")
    parser = argparse.ArgumentParser(description='persona_vocab')
    parser.add_argument('SOURCE', help='source sentence list Ken : I \' mのほう')

    args = parser.parse_args()
    
    with io.open('persona_vocab.en', 'w') as wf:
        bar = progressbar.ProgressBar()
        persona = []
        with io.open(args.SOURCE, 'r', encoding='utf-8', errors='ignore') as rf:
            for line in rf:
                parson = re.match(r'[^(:|\n)]*:', line)
                if parson != None:
                    parson = parson.group(0)[:-1].strip()
                else:
                    parson = "none"
                
                persona.append(parson.lower())
            #print(persona)
            persona = list(set(persona))
            #print("------------------")
            #print(persona)
        
        count = 0
        for line in persona:
            if line is not None and line != '':
                print(line)
                wf.write(line)
                wf.write('\n')
                count += 1
        print(count)


if __name__ == '__main__':
    main()


