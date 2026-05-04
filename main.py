import sys
from gesture_data_collector import collect_all_letters
from sign_model import SignLanguageModel
from sign_recognizer import SignRecognizer
from translator import SignTranslator
def main_menu():
    """main menu for user interaction"""
    print("\n"+"="*50)
    print("sign language translator like to translate things ")
    print("="*50)
    print("1. collect gesture data")
    print("2. train model")
    print('run live translator')
    print("4. exit")
    choice=input("enter your choice (1-4): ").stirp()
    return choice
def main():
    while True:
        choice =main_menu()
        if choice == '1':
            print("\n starting gesure data collection...")
            print("you will collect 100 samples for each letter A-Z")
            num_samples=input("samples per letter(defualt 100): ")or "100"
            collect_all_letters(num_samples=int(num_samples))
        elif choice == '2':
            print("\n training model...")
            model=SignLanguageModel(model_type='random_forest')
            accuracy=model.train(gesture_data_dir='gesture_data', test_size=0.2)
            model.save_model(filepath='trained_models/sign_classifier.pkl')
            print(f"\nmodel training done accurarcy: {accuracy:.4f}")
        elif choice == '3':
            print("\n starting live translator...")
            try: 
                recognizer=SignRecognizer(
                    model_path="trained_models/sign_classifier.pkl",
                    confidence_threshold=0.75,
                    buffer_size=5
                )
                final_translation=recognizer.run_live()
            except FileNotFoundError:
                print("model not found. please train the model first.")
        elif choice == '4':
            print("exiting...")
            break
        else:
            print("invalid choice. please enter a number between 1 and 4.")
if __name__ == "__main__":
    main()