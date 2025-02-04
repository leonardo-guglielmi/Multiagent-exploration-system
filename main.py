from datetime import datetime as date

from Simulate import simulate
from User import User
from Constants import *
from Sensor import Sensor

def experiment1():

    try:
        types_of_search = ["local"]
        expl_weights = ["constant"]

        print(f"Simulations begin: {date.now()}\n")
        with open("logs/output_log.txt", 'w') as f:
            f.write(f"Simulations begin: {date.now()}\n")

        for i in range(NUM_OF_SIMULATIONS):
            deserialize = False
            for type_of_search in types_of_search:
                for expl_weight in expl_weights:
                    for expl  in [True, False]:
                        for BS in [True, False]:
                            print(f'----- Starting simulation [{type_of_search}-{expl_weight}-no expl] : {i} @ {date.now()}-----')
                            with open("logs/output_log.txt", 'a') as f:
                                f.write(f'----- Starting simulation [{type_of_search}-{expl_weight}-no expl] : {i} @ {date.now()}-----\n')
                            simulate(type_of_search, expl_weight, i, deserialize, use_expl=expl, use_bs=BS)
                            Sensor.id = 0
                            User.id = 0
                            deserialize = True

        print(f"Simulations completed {date.now()}")
        with open("logs/output_log.txt", 'a') as f:
            f.write(f"Simulations completed {date.now()}\n")

    except Exception as e:
        with open("logs/error_log.txt", "w") as f:
            f.write(str(e))
        with open("logs/output_log.txt", 'a') as f:
            f.write(str(e))
        raise e

def experiment2():
    try:
        types_of_search = ["systematic", "local", "annealing forward", "annealing reverse", "penalty"]
        expl_weights = ["constant"]

        print(f"Simulations begin: {date.now()}\n")
        with open("logs/output_log.txt", 'w') as f:
            f.write(f"Simulations begin: {date.now()}\n")

        for i in range(NUM_OF_SIMULATIONS):
            deserialize = False
            for type_of_search in types_of_search:
                for expl_weight in expl_weights:
                    print(f'----- Starting simulation [{type_of_search}-{expl_weight}] : {i} @ {date.now()}-----')
                    with open("logs/output_log.txt", 'a') as f:
                        f.write(f'----- Starting simulation [{type_of_search}-{expl_weight}] : {i} @ {date.now()}-----\n')

                    simulate(type_of_search, expl_weight, i, deserialize, use_expl=True, use_bs=True)
                    Sensor.id = 0
                    User.id = 0
                    deserialize = True
        print("Simulations completed")
        with open("logs/output_log.txt", 'a') as f:
            f.write("Simulations completed\n")

    except Exception as e:
        with open("logs/error_log.txt", "w") as f:
            f.write(str(e))
        with open("logs/output_log.txt", 'a') as f:
            f.write(str(e))
        raise e

def experiment3():
    try:
        types_of_search = ["local"]
        expl_weights = ["constant", "decrescent"]

        for i in range(NUM_OF_SIMULATIONS):
            deserialize = False
            for type_of_search in types_of_search:
                for expl_weight in expl_weights:
                    print(f'----- Starting simulation [{type_of_search}-{expl_weight}] : {i} @ {date.now()}-----')
                    with open("logs/output_log.txt", 'a') as f:
                        f.write(f'----- Starting simulation [{type_of_search}-{expl_weight}] : {i} @ {date.now()}-----\n')

                    simulate(type_of_search, expl_weight, i, deserialize, use_expl=True, use_bs=True)
                    Sensor.id = 0
                    User.id = 0
                    deserialize = True
        print("Simulations completed")
        with open("logs/output_log.txt", 'a') as f:
            f.write("Simulations completed\n")

    except Exception as e:
        with open("logs/error_log.txt", "w") as f:
            f.write(str(e))
        with open("logs/output_log.txt", 'a') as f:
            f.write(str(e))
        raise e

if __name__ == '__main__':
    experiment3()
