import argparse
import re

def main():
    parser = argparse.ArgumentParser(description="File Path")
    parser.add_argument("path", help="input file")

    args = parser.parse_args()

    with open(args.path, "r") as f:
        inputRead = f.read().split("\n")
    
        print("input persona!")
        persona = input()
        persona = persona + " :"
        output = ""
        for i in range(len(inputRead)):
            line = re.search(persona, inputRead[i])
            if  line != None:
                output += inputRead[i-1] + "\n" + inputRead[i] + "\n\n"
        
    with open("personaSentence.txt", "w") as f:
        f.write(output)






if __name__ == "__main__":
    main()