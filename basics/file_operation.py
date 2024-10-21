# Create the txt file
def write_to_file():
    with open("data.txt","w",encoding='utf-8') as file:
        input_line=input("Enter text(type 'q' to exit): ")
        while input_line.lower() !='q':
            file.write(input_line+" \n")
            input_line = input("Enter text (type 'q' to exit): ")  # Ask for input again

def read_and_analyze_file():
    try:
        with open("data.txt","r",encoding="utf-8") as file:
            content=file.read()
            print(content)
            char_count=len(content)
            line_count=len(content.splitlines())
            word_count=len(content.split(" "))-1 #"/n " at EOF
            print(f'Character:{char_count}, Words:{word_count}, Lines:{line_count}')
    except FileNotFoundError:
        print("file not found.")
    except PermissionError:
        print("Permission denied")


if __name__=="__main__":
    write_to_file()
    read_and_analyze_file()


