import json
import random
import string
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import os
import argparse

# Download necessary NLTK data
nltk.download('punkt', quiet=True)  # Download necessary NLTK data
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()


# Load data from file
def load_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        print(f"Successfully loaded data from {file_path}")
        return data
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: {file_path} is not a valid JSON file.")
        exit(1)
    except Exception as e:
        print(f"An error occurred while loading the file: {str(e)}")
        exit(1)


def preprocess_data(data):
    words = []
    classes = []
    documents = []
    ignore_letters = ['?', '!', '.', ',']

    # Extract all patterns and tags
    for intent in data['intents']:
        for pattern in intent['patterns']:
            # Tokenize each word in the pattern
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            # Add documents in the corpus
            documents.append((word_list, intent['tag']))
            # Add to our classes list
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    # Lemmatize words and remove duplicates
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
    words = sorted(list(set(words)))
    classes = sorted(list(set(classes)))

    print(f"Preprocessed {len(documents)} documents")
    print(f"Found {len(classes)} classes: {classes}")
    print(f"Unique lemmatized words: {len(words)}")

    return words, classes, documents


def create_training_data(words, classes, documents):
    # Create training data
    training = []
    # Create an empty array for our output
    output_empty = [0] * len(classes)

    # Training set, bag of words for each pattern
    for doc in documents:
        # Initialize our bag of words
        bag = []
        # List of tokenized words for the pattern
        pattern_words = doc[0]
        # Lemmatize each word to create the base word
        pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

        # Create our bag of words array
        for w in words:
            bag.append(1) if w in pattern_words else bag.append(0)

        # Output is a '0' for each tag and '1' for current tag
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1

        training.append([bag, output_row])

    # Shuffle our features and convert to numpy arrays
    random.shuffle(training)
    training = np.array(training, dtype=object)

    # Create train and test lists
    train_x = list(training[:, 0])
    train_y = list(training[:, 1])

    print(f"Created training data with {len(train_x)} samples")

    return np.array(train_x), np.array(train_y)


# Build and train the model
def create_model(input_shape, num_classes):
    # Create model - simple neural network
    model = tf.keras.Sequential([
        Dense(128, input_shape=(input_shape,), activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    # Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def build_and_train_model(train_x, train_y, epochs=200, batch_size=5):
    input_shape = len(train_x[0])
    num_classes = len(train_y[0])

    model = create_model(input_shape, num_classes)

    # Train model
    print("Training model...")
    history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=1)
    print("Model trained!")

    # Print final accuracy
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")

    return model


# Prediction functions
def clean_up_sentence(sentence):
    # Tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # Lemmatize each word - create base word, in attempt to represent related words
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# Return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words):
    # Tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # Bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # Assign 1 if current word is in the vocabulary position
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence, model, words, classes, ERROR_THRESHOLD=0.25):
    # Filter out predictions below a threshold
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]

    # Get top prediction
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    # Sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})

    return return_list


def get_response(ints, intents_json):
    if not ints:
        return "I'm not sure I understand. Could you please rephrase that?"

    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']

    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break

    return result


# Save and load functions
def save_model_and_data(model, words, classes, output_dir='model_output'):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_path = os.path.join(output_dir, 'model.h5')
    words_path = os.path.join(output_dir, 'words.pkl')
    classes_path = os.path.join(output_dir, 'classes.pkl')

    model.save(model_path)

    with open(words_path, 'wb') as words_file:
        pickle.dump(words, words_file)

    with open(classes_path, 'wb') as classes_file:
        pickle.dump(classes, classes_file)

    print(f"Model and data saved to {output_dir} directory")
    return model_path, words_path, classes_path


def load_model_and_data(model_dir='model_output'):
    model_path = os.path.join(model_dir, 'model.h5')
    words_path = os.path.join(model_dir, 'words.pkl')
    classes_path = os.path.join(model_dir, 'classes.pkl')

    if not os.path.exists(model_path) or not os.path.exists(words_path) or not os.path.exists(classes_path):
        print(f"Error: Model files not found in {model_dir}")
        return None, None, None

    model = tf.keras.models.load_model(model_path)

    with open(words_path, 'rb') as words_file:
        words = pickle.load(words_file)

    with open(classes_path, 'rb') as classes_file:
        classes = pickle.load(classes_file)

    print(f"Model and data loaded from {model_dir}")
    return model, words, classes


# Chatbot class to put it all together
class MentalHealthChatbot:
    def __init__(self, data_path, model_dir='model_output', load_existing=False, epochs=200):
        self.intents = load_data(data_path)
        self.model_dir = model_dir

        if load_existing and os.path.exists(os.path.join(model_dir, 'model.h5')):
            print("Loading existing model...")
            self.model, self.words, self.classes = load_model_and_data(model_dir)
            if self.model is None:
                print("Failed to load existing model. Training new model...")
                self._train_new_model(epochs)
        else:
            print("Generating new model...")
            self._train_new_model(epochs)

    def _train_new_model(self, epochs):
        # Preprocess data
        self.words, self.classes, documents = preprocess_data(self.intents)

        # Create training data
        train_x, train_y = create_training_data(self.words, self.classes, documents)

        # Build and train the model
        self.model = build_and_train_model(train_x, train_y, epochs=epochs)

        # Save the model and data
        save_model_and_data(self.model, self.words, self.classes, self.model_dir)

    def get_response(self, message):
        # Predict the intent
        ints = predict_class(message, self.model, self.words, self.classes)

        # Get a response
        return get_response(ints, self.intents)

    def chat(self, message):
            message = message

            response = self.get_response(message)
            return response


# Main function with argument parsing
def main():
    parser = argparse.ArgumentParser(description="Mental Health Chatbot")
    parser.add_argument("--data_path", type=str,
                        help="ePath to the JSON dataset file", default="KB.json")
    parser.add_argument("--model_dir", type=str, default="model_output",
                        help="Directory to save the generated model")
    parser.add_argument("--load_existing", action="store_true",
                        help="Load existing model if available (optional)")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of training epochs")

    args = parser.parse_args()

    print(args.load_existing)

    # Create and start the chatbot
    print(f"Initializing Mental Health Chatbot with data from {args.data_path}")
    chatbot = MentalHealthChatbot(
        data_path=args.data_path,
        model_dir=args.model_dir,
        load_existing=args.load_existing,
        epochs=args.epochs
    )
    chatbot.chat()


if __name__ == "__main__":
    main()