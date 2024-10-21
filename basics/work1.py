import random
class InvalidInputError(Exception):
    # Age smaller than0 or greater than 120
    # def __init__(self):
    #     return
    pass
def work1():
    integer=10
    float=0.2
    string="hello"
    boolean=True
    print(f'typpe of integer:{type(integer)},value:{integer}')
    print(f'typpe of string:{type(string)},value:{string}')
    print(f'typpe of float:{type(float)},value:{float}')
    print(f'typpe of boolean:{type(boolean)},value:{boolean}')

def work2():
    example="life is short, i use python"
    print(example)
    print(example.upper())
    example2=example.upper()
    print(example2.lower())
    print(example.split(","))

def grade():
    score=float(input("Enter your score (1-100):"))
    if score>=90:
        grade_class="A"
    elif score>=80:
        grade_class="B"
    elif score>=70:
        grade_class="C"
    elif score>=60:
        grade_class="D"
    else:
        grade_class="F"
    print(f"Your grade is:{grade_class}")

def calculate():
    input_calculation=input("Enter calculation:")
    if "+" in input_calculation:
        num1,num2=input_calculation.split("+")
        result=float(num1)+float(num2)
    elif "-" in input_calculation:
        num1,num2=input_calculation.split("-")
        result=float(num1)-float(num2)
    elif "*" in input_calculation:
        num1,num2=input_calculation.split("*")
        result=float(num1)*float(num2)
    elif "/" in input_calculation:
        num1,num2=input_calculation.split("/")
        result=division_calculation(num1,num2)
    else:
        print("Invalid operation")
    print(f"The result of {input_calculation} is:{result}")

def print_prime_numbers():
    prime_nums=[]
    for num in range(2,101):
        num_prime=True
        for i in range(2,int(num**0.5)+1):
            if num%i==0:
                num_prime=False
                break
        if num_prime:
            prime_nums.append(num)
    print(prime_nums)

def guess_number():
    random_number=random.randint(1,100)
    user_guess=-1
    while user_guess!=random_number:
        user_guess=int(input("Enter your guess:"))
        if user_guess<random_number:
            print("Too low")
        elif user_guess>random_number:
            print("Too high")
    print(f"Congratulations! You guessed the number {random_number} correctly!")

def fav_book_list():
    fav_books=[]
    for i in range(5):
        fav_books.append(input(f"Enter your favorite book {i+1}:"))
    print(fav_books)
    fav_books.remove(input("Enter the book you want to remove:"))
    print(fav_books)
    print("removing the 3rd book")
    fav_books.pop(2)
    print(fav_books)
    index=int(input("Enter the index of the book you want to edit:"))-1
    fav_books[index]=input("Enter the new book name:")
    print(fav_books)

def friend_dict():
    # create a dictionary
    friend_info={
        "name":[],
        "age":[],
        "job":[]
    }
    for i in range(1):
        friend_info["name"].append(input(f"Enter friend {i+1} name:"))
        friend_info["age"].append(int(input(f"Enter friend {i+1} age:")))
        friend_info["job"].append(input(f"Enter friend {i+1} job:"))
    print(friend_info)

    # #access the dictionary
    # print(friend_info["age"])
    # for details in friend_info.items():
    #     if details["age"]==30:
    #         print(f"{details["name"]}")


    print(friend_info.get("first_name","Error"))

def division_calculation(num1,num2):
    try:
        result=num1/num2
        print(f"result:{result}")
    except ZeroDivisionError:
        print("Cannot divide by 0")
    except ValueError:
        print("Invalid input")

def check_age(age):
    if age<0 or age>120:
        raise InvalidInputError(f"Age {age} is not between 0 to 120")
    

if __name__=='__main__':
    try:
        age=int(input("Enter your age: "))
        check_age(age)
    except InvalidInputError as e:
        print(f"Invalid input error:{e}")
    except ValueError:
        print("Input a valid integer for age")

