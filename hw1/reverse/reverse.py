import os
import sys

def main():
    # check for empty file
    if(os.stat(str(sys.argv[1])).st_size == 0):
        open(str(sys.argv[2]), "w").close()

    else:
        """
            1. open file
            2. strip lines and reverse them
            3. convert to string and write to file
        """
        with open(str(sys.argv[1]), "r") as infile:
            infile_lines = infile.readlines()
        
        lines = [x.strip() for x in infile_lines]
        outlines = "\n".join(lines[::-1])
        outlines = outlines+"\n"
        with open(str(sys.argv[2]), "w") as outfile:
            outfile.write(outlines)

        infile.close()
        outfile.close()

if  __name__ == "__main__":
    main()