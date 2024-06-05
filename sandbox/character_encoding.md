# What's character encoding?

## Links
[Real Python](https://realpython.com/python-encodings-guide/#one-byte-two-bytes-three-bytes-four)

It encompasses the following:

- Lowercase English letters: a through z
- Uppercase English letters: A through Z
- Some punctuation and symbols: "$" and "!", to name a couple
- Whitespace characters: an actual space (" "), as well as a newline, carriage return, horizontal tab, vertical tab, and a few others
- Some non-printable characters: characters such as backspace, "\b", that can’t be printed literally in the way that the letter A can


So what is a more formal definition of a character encoding?

At a very high level, it’s a way of translating characters (such as letters, punctuation, symbols, whitespace, and control characters) to integers and ultimately to bits. Each character can be encoded to a unique sequence of bits. Don’t worry if you’re shaky on the concept of bits, because we’ll get to them shortly.

The various categories outlined represent groups of characters. Each single character has a corresponding code point, which you can think of as just an integer. Characters are segmented into different ranges within the ASCII table: