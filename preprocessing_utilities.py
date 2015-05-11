__author__ = 'pat'
import os
import shutil
import numpy as np
import math
import inflect
from itertools import groupby
import translitcodec            # For unicode to ASCII text sanitization






def str_to_seq(strng):
    seq = []
    for c in strng:
        val = ord(c)
        if val==32: val = 26
        elif val==39: val=27
        elif val==45: val=28
        else: val-=97
        seq.append(val)
    return seq


def seq_to_str(seq):
    strng = ''
    for elem in seq:
        if elem==26: strng += ' '
        elif elem==27: strng += '\''
        elif elem==28: strng += '-'
        elif elem==29: pass
        else: strng += chr(elem+97)
    return strng


# Remove consecutive symbols and blanks
def F(pi, blank):
    return [a for a in [key for key, _ in groupby(pi)] if a != blank]


# Insert blanks between unique symbols, and at the beginning and end
def make_l_prime(l, blank):
    result = [blank] * (len(l) * 2 + 1)
    result[1::2] = l
    return result


def send_label_to_trainer(l, blank):
    label = str_to_seq(l)
    label = F(label, blank)
    return make_l_prime(label, blank)

def good_path(probs, blank=29):
    out = []
    for timestep in probs:
        if timestep[blank]<.7:
            tmp = blank
            tmp_max = 0.0
            for i in np.arange(len(timestep[:-1])):
                if timestep[i]>tmp_max and timestep[i] > .2:
                    tmp = i
                    tmp_max = timestep[i]
            out.append(tmp)
            #out.append(np.argmax(timestep[:-1]))
        else:
            out.append(np.argmax(timestep))
    return out




def log_add_logs(loga, logb):
    return loga + np.log(1 + np.exp(logb - loga))

















# Get the directories of the stories:
# Returns a list of directory paths of the form "root_dir/story_dir"
def get_story_paths(root_dir):
    assert(os.path.isdir(root_dir))
    return [os.path.join(root_dir, dirr) for dirr in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, dirr))]


# Get the directories of all of the story segments
# Returns a list of directory paths of the form "root_dir/story_dir/segment_dir"
def get_story_segment_paths(root_dir=None, list_of_story_paths=None):
    assert(root_dir is not None or list_of_story_paths is not None)

    if root_dir is not None:
        list_of_story_paths = get_story_paths(root_dir)

    segment_directories = []
    for story_directory in list_of_story_paths:
        for segment_directory in [dirr for dirr in os.listdir(story_directory) if os.path.isdir(os.path.join(story_directory, dirr))]:
            segment_directories.append(os.path.join(story_directory, segment_directory))
    return segment_directories


# Partition any WAV files in the directory structure that have not yet been partitioned
def partition_audio_files(root_dir, N=50):
    assert(os.path.isdir(root_dir))

    segment_directories = get_story_segment_paths(root_dir=root_dir)

    for segment_directory in segment_directories:
        # get a list of the wav files
        wav_files = [fname for fname in os.listdir(segment_directory) if fname.lower().endswith('.wav')]
        wav_files.sort(key=lambda fname: int(os.path.splitext(fname)[0].split('-')[-1]))

        # check if some files have already been partitioned
        existing_dirs = [dirr for dirr in os.listdir(segment_directory) if os.path.isdir(os.path.join(segment_directory, dirr))]
        existing_dirs_integers = []
        for dirr in existing_dirs:
            try:
                existing_dirs_integers.append(int(dirr))
            except: pass
        if existing_dirs_integers == []: starting_index = 0
        else: starting_index = int(max(existing_dirs_integers))+1

        # Create the directories and move the files
        partition = 0
        for i in np.arange(starting_index, int(math.ceil(len(wav_files)/float(N)))+starting_index):
            os.mkdir(os.path.join(segment_directory, str(i)))

            for wav_file in wav_files[partition*N:(partition+1)*N]:
                shutil.move(os.path.join(segment_directory, wav_file), os.path.join(segment_directory, str(i), wav_file))

            partition += 1


def get_story_segment_partition_paths(root_dir=None, list_of_story_paths=None, list_of_segment_paths=None):
    assert(root_dir is not None or list_of_story_paths is not None or list_of_segment_paths is not None)

    if root_dir is not None:
        list_of_segment_paths = get_story_segment_paths(root_dir=root_dir)
    elif list_of_story_paths is not None:
        list_of_segment_paths = get_story_segment_paths(list_of_story_paths=list_of_story_paths)

    partition_paths = []
    for segment_path in list_of_segment_paths:
        for partition_folder in [dirr for dirr in os.listdir(segment_path) if os.path.isdir(os.path.join(segment_path, dirr))]:
            try:
                # make sure the folder is an integer name
                int(partition_folder)

                partition_paths.append(os.path.join(segment_path, partition_folder))
            except:
                pass

    return partition_paths


def sanitize_text(text):
    # A list of strings to replace
    replace_tuples = [('&c', 'etcetera'), ('&', 'and'), ('=', 'equal')]
    for replace_tuple in replace_tuples:
        text = text.replace(replace_tuple[0], replace_tuple[1])


    # This is not an exhaustive list of forbidden characters; we need to implement a unicode -> "close ASCII" converter.
    bad_characters_remove = ['.', ',', ':', '?', '_', '!', '"', ';', "'", '*', '(', ')', '/', '\\']
    bad_characters_replace_with_space = ['-', '\t']

    text = text.lower().replace('\r\n', ' ').replace('\n', ' ')
    text = text.translate(None, ''.join(bad_characters_remove))
    for char in bad_characters_replace_with_space: text = text.replace(char, ' ')

    # Numerals... Ah, why can't we just say one zero three instead of one hundred and three or many of the variations.
    # It may be best to replace these, and then invalidate these in some way, later, by default
    output_text = []
    numerals = [str(x) for x in np.arange(10)]
    for word in text.split(' '):
        found_numeral = False
        for numeral in numerals:
            if numeral in word:
                found_numeral = True
                if '/' in word:
                    if word == '1/2': output_text.append('one half')
                    # else, use the 1/4 -> one-fourth paradigm
                    else:
                        fraction = word.split('/')
                        output_text.append(inflector.number_to_words(fraction[0]))
                        output_text.append(inflector.number_to_words(inflector.ordinal(fraction[1])))
                    break
                else:
                    output_text.append(inflector.number_to_words(word))
                    if '$' in word:
                        output_text.append('dollars')
                    break
        if not found_numeral:
            output_text.append(word)

    text = ' '.join(output_text)
    text = text.translate(None, ''.join(bad_characters_remove))
    for char in bad_characters_replace_with_space: text = text.replace(char, ' ')

    # Unicode to ASCII/english spellings
    text = unicode(text, encoding='utf-8').encode('translit/long/ascii', 'replace')

    # replace double spaces
    while '  ' in text:
        text = text.replace('  ', ' ')

    return text.replace('\n', ' ').replace('\t', ' ').strip()


def sanitize_partitioned_text(root_dir):
    partition_paths = get_story_segment_partition_paths(root_dir=root_dir)

    for partition_path in partition_paths:
        if partition_path.endswith('/'):
            text_file_name = partition_path.split('/')[-2]
        else:
            text_file_name = partition_path.split('/')[-1]

        text_path = os.path.join(partition_path, text_file_name) + '.txt'

        overwrite_original = True
        if os.path.exists(text_path.replace('.txt', '__original.txt')):
            overwrite_original = False

        if os.path.exists(text_path):
            text = ''
            # if there is not an original text, open the #.txt file
            if overwrite_original:
                with open(text_path, 'r') as f:
                    text = f.read()
                text = sanitize_text(text)

                # rename the original file
                new_fname = text_path.replace('.txt', '__original.txt')
                shutil.move(text_path,  new_fname)

            # Else grab the  original text to sanitize from scratch again
            else:
                with open(text_path.replace('.txt', '__original.txt'), 'r') as f:
                    text = f.read()
                text = sanitize_text(text)

            # write back
            with open(text_path, 'w') as f:
                f.write(text)



inflector = inflect.engine()
def anomalies_in_text(root_dir):
    partition_paths = get_story_segment_partition_paths(root_dir=root_dir)

    for partition_path in partition_paths:
        if partition_path.endswith('/'):
            text_file_name = partition_path.split('/')[-2]
        else:
            text_file_name = partition_path.split('/')[-1]

        text_path = os.path.join(partition_path, text_file_name) + '.txt'

        if os.path.exists(text_path):

            text = ''
            # read and sanitize
            with open(text_path, 'r') as f:
                text = f.read()
            text = sanitize_text(text)

            for word in text.split(' '):
                try:
                    converted = send_label_to_trainer(word, 29)

                    # try to break it
                    tester = [x for x in np.arange(30)] # array of possible outputs, although I'm pretty sure np.arange is an array

                    for index in converted:
                        tester[index]
                except:
                    #print sys.exc_info()
                    print word, inflector.number_to_words(word), [ord(c) for c in unicode(word, encoding='utf-8').encode('translit/long/ascii', 'replace')]