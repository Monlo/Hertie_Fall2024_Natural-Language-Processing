# ---------------------------------------------------------------------------------------
# NATURAL LANGUAGE PROCESSING
# Session 2: Regular Expressions
# Instructor: Sascha Göbel
# September 2024
# ---------------------------------------------------------------------------------------


# PREPARATIONS ==========================================================================

# load libraries ------------------------------------------------------------------------
import re
import pandas as pd
import os

# Change directory ---------------------------------------------------------------------------
os.chdir('01_Regular Expressions/')

# import data ---------------------------------------------------------------------------
star_trek_bios = open("star-trek-bios.txt").read()
print(star_trek_bios)

# BASIC STRING OPERATIONS ===============================================================

# assign a string to an object and print ------------------------------------------------
a_string = ("  Line 1: Python    string objects record textual information.\n"
            "Line 2: They collect   one-character strings in a sequence,\n"
            "Line 3: maintaining \t left to right   order.\n")
print(a_string)

# count characters in a string (extraction) ---------------------------------------------
len(a_string)

# extract from a string by position (extraction) ----------------------------------------
a_string[0]
a_string[2:8]
a_string[-1]

# concatenating and repeating strings (modification) ------------------------------------
print(a_string + "Line 4: ... ")
print(a_string + ("\nNew line" * 5))

# convert to lower-case and upper-case (modification) -----------------------------------
print(a_string.lower())
print(a_string.upper())

# remove spaces from beginning and end of a string (modification) -----------------------
print(a_string.strip())

# pad strings (modification) ------------------------------------------------------------
help(a_string.center)
a_string.center(len(a_string)+4, "-")

# count and find substrings (extraction) ------------------------------------------------
a_string.count("Line") # Counts the appearances of the string
a_string.find("Line 2") #

# replace substrings (modification) -----------------------------------------------------
a_string[63:69] = ""  # doesn't work, strings are immutable
print(a_string.replace("Line 2", ""))

# split and join strings (splitting) ----------------------------------------------------
a_string.split()
a_string.splitlines()

# remove redundant white space, tabs, and new lines (modification) ----------------------
print(" ".join(a_string.split()))

# REGULAR EXPRESSIONS ===================================================================

# compiling or assigning patterns -------------------------------------------------------
pattern_1 = re.compile(pattern="\\d")  # alternatively re.compile(r"\d")
pattern_1_1 = re.compile(pattern= r"\d")

pattern_2 = re.compile(pattern="Line \\d:|\\s+")  # alternatively re.compile(r"Line \d| +|\t|\n")
pattern_2_2 = re.compile(r"Line \d| +|\t|\n")
pattern_2_alt = "Line \\d:|\\s+"

# find first match for a pattern --------------------------------------------------------
match_1 = pattern_1.search(string=a_string)
match_1.group()

# find all matches for a pattern --------------------------------------------------------
pattern_1.findall(string=a_string)
pattern_2.findall(string=a_string)
re.findall(pattern=pattern_2_alt, string=a_string)

# replace all matches for a pattern -----------------------------------------------------
re.sub(pattern=pattern_2, repl=" ", string=a_string)

# examples ------------------------------------------------------------------------------
# split bios into separate strings
star_trek_bios = star_trek_bios.split(sep="\n\n")
star_trek_bios.pop(0)
print(star_trek_bios[0])

# extract names
names = [re.search(pattern="^.+?,", string=i).group() for i in star_trek_bios]

# The regular expression ^.+?, can be broken down as follows:
# ^: Asserts the position at the start of the string.
# .: Matches any character (except newline by default).
# +?: Matches one or more of the preceding character (any character in this case), but in a non-greedy (or lazy) manner, meaning it matches the smallest possible number of characters that still allows the rest of the pattern to succeed.
# ,: Matches a literal comma character.

print(names)
names = [re.sub(pattern=",", repl="", string=i) for i in names]
print(names)

# extract birth dates and places
dates_and_places = [re.search(pattern="born on.+?,.+?,.+?, (?:[A-Z]\\w+)?", string=i).group() for i in star_trek_bios]
# extracts both dates and places ensuring that if no second place mention exists, it does not go on
# by requiring the second mention to start with a capital letter, ?: makes the group non-capturing
dates = [re.search(pattern="[A-Z]\\w+ \\d{1,2}, \\d{4}", string=i).group() for i in dates_and_places]
places = [re.search(pattern="(?<=, (?:in|on)).+", string=i).group() for i in dates_and_places]
# lookbehind

# extract starfleet service numbers
service_numbers = [re.search(pattern="\\d{3}-\\d{3}-\\d{3}", string=i).group() for i in star_trek_bios]

# email
emails = [re.search(pattern="(?:\\w+\\.)?\\w+@\\w+\\.\\w+", string=i).group() for i in star_trek_bios]

# position
positions = [re.search(pattern="(?<=\\w was ).+?(?=(?: of | on ))", string=i).group() for i in star_trek_bios]
# \\w in the beginning to ensure that ", was born" is not included
positions = [re.sub(pattern="^.+?later (?:the)?|^the |^an", repl="", string=i) for i in positions]

# vessel
vessels = [re.search(pattern="USS.+?\\)|Deep Space Nine", string=i).group() for i in star_trek_bios]

# psych profile
profiles = [re.search(pattern="(?<=profile(?! is)).+$", string=i).group() for i in star_trek_bios]
# nested lookahead "profile(?! is)", profile that is not followed by is (indicates a social media account)
# nested in (?<=...).+$) everything following after the eligible "profile".

# assemble in pandas data frame
data = {
    "name": names,
    "dob": dates,
    "pob": places,
    "service_number": service_numbers,
    "position": positions,
    "vessel": vessels,
    "email": emails,
    "profile": profiles
}
df = pd.DataFrame(data)
