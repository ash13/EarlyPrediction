import datautility as du
import numpy as np
import pandas as pd
import sys
import csv
import time


def ind(headers, name, case_sensitive=False):
    """
    function uses a list of headers and a column name and returns the index of that column

    :param headers: a 1-D list of column names
    :param name: a string of the column name to find
    :param case_sensitive: boolean value (converts names to lowercase for comparison if false)
    :return: the index of the corresponding column name or None if not found
    """
    try:
        if not case_sensitive:
            return np.argwhere(np.array([str(i).lower() for i in headers], dtype=str) == str(name).lower()).ravel()[0]
        else:
            return np.argwhere(np.array(headers, dtype=str) == str(name)).ravel()[0]
    except IndexError:
        return None


def generate_features(data_csv):
    """
    function uses the data csv file to:
     - load, filter, and sort the data
     - write the filtered data to a temporary file
     - read the filtered data line by line to find each student
     - generate the features for each student and write to file

    :param data_csv: the file name of the data csv
    :return: None (writes [data_csv]_features.csv
    """

    # the name of the ordered, filtered datafile (will create if doesn't exist)
    filename = 'ordered_sb_16_17.csv'
    outfile = data_csv[:-4] + '_features.csv'

    # if the above file is not found, we will use pandas to load, filter, and sort the data
    if not np.any([f.find(filename) >= 0 for f in du.getfilenames(extension='.csv')]):
        # read the data csv using pandas
        data = pd.read_csv(data_csv)
        print("original data length:", len(data.index))

        # filter out the unneeded action rows
        data = data[~data['action_name'].isin(['start', 'comment', 'end', 'work', 'resume', '0'])]

        # sort by user, assignment, problem, and action and write the new data to file
        data.sort_values(by=['user_id',
                             'first_problem_log',
                             'problem_log_id',
                             'action_time']).to_csv(filename,index=False)

        # print the length of the new filtered data (should be smaller than the original)
        print("filtered data length:", len(data.index))

        # release the memory for the pandas dataframe, it is no longer needed
        data = None

    # the following will open the file and read line by line, generating the features for each student
    # this avoids the need to keep the data in memory as it loads only a single student at a time
    n_lines = len(open(filename).readlines())
    with open(filename, 'r', errors='replace') as f:
        # find the number of lines in the file to print progress
        f_lines = csv.reader(f)
        output_str = '-- loading {}...({}%)'.format(filename, 0)
        sys.stdout.write(output_str)
        sys.stdout.flush()
        old_str = output_str
        i = 0

        # initialize variables for later use
        student = None
        h = None
        header_out = None
        written = False
        write_timout = 60

        # for every line in the file...
        for line in f_lines:
            if len(line) == 0:
                continue

            # clean up any NA values that may exist from excel or other programs (such as R)
            line = np.array(line)
            na = np.argwhere(np.array(line[:]) == '#N/A').ravel()
            if len(na) > 0:
                line[na] = ''
            na = np.argwhere(np.array(line[:]) == 'NA').ravel()
            if len(na) > 0:
                line[na] = ''

            # if this is the first line, grab the column names and add the new feature column names for later
            if i == 0:
                h = line
                header_out = list(h)
                header_out.extend(['used_penultimate_hint', 'used_bottom_out_hint', 'previous_3_actions',
                                   'current_and_past_2_actions', 'next_assignment_wheel_spin',
                                   'next_assignment_stopout'])
                i += 1
                continue

            # look for each student's data and stop to process once we have all of one student's rows
            if student is None:
                student = line.reshape((-1,len(h)))
            elif student is not None and line[ind(h,'user_id')] == student[-1, ind(h, 'user_id')]:
                student = np.append(student, line.reshape((-1,len(h))), axis=0)
            else:
                # we have all of the student's rows, now we generate features
                _, a = np.unique(student[:, ind(h,'assignment_id')], return_index=True)

                # get the assignment-level rows and offset wheel spinning and stopout
                next_ws = np.append(student[a, ind(h, 'assignment_id')].reshape((-1,1)),
                                    np.append(student[a, ind(h, 'assignment_wheel_spin')].reshape((-1)),
                                              np.nan)[1:].reshape((-1,1)), axis=1)
                next_so = np.append(student[a, ind(h, 'assignment_id')].reshape((-1, 1)),
                                    np.append(student[a, ind(h, 'assignment_stopout')].reshape((-1)),
                                              np.nan)[1:].reshape((-1, 1)), axis=1)

                # get each of the assignments while maintaining assignment order
                assignments = student[a, ind(h, 'assignment_id')]
                for asn in assignments:
                    st_assignment = student[np.argwhere(np.array(student[:,ind(h, 'assignment_id')]) == asn).ravel(), :]

                    # get each of the problems within the assignment while maintaining problem order
                    _, p = np.unique(st_assignment[:, ind(h, 'problem_log_id')], return_index=True)
                    problems = st_assignment[p, ind(h, 'problem_log_id')]
                    for pr in problems:
                        # subset to all actions for the problem to generate within-problem features
                        st_problem = st_assignment[np.argwhere(np.array(st_assignment[:,
                                                                        ind(h, 'problem_log_id')]) == pr).ravel(), :]
                        # replace missing/null values with a default value of 0
                        st_problem[np.where(st_problem == '')] = 0

                        # cumulative use of penultimate hint
                        penultimate_hint = np.array(np.array(np.array(
                            np.cumsum(
                                [int(a.find('hint') >= 0) for a in st_problem[:, ind(h, 'action_name')]])) >= float(
                            st_problem[0, ind(h, 'problem_total_hints')]) - 1, dtype=np.int32), dtype=str).reshape(
                            (-1, 1)) if float(st_problem[0, ind(h, 'problem_total_hints')]) > 0 else np.zeros(
                            (len(st_problem), 1))

                        # cumulative use of bottom out hint
                        bottom_hint = np.array(np.array(np.array(
                            np.cumsum(
                                [int(a.find('hint') >= 0) for a in st_problem[:, ind(h, 'action_name')]])) >= float(
                            st_problem[0, ind(h, 'problem_total_hints')]), dtype=np.int32), dtype=str).reshape(
                            (-1, 1)) if float(st_problem[0, ind(h, 'problem_total_hints')]) > 0 else np.zeros(
                            (len(st_problem), 1))

                        # the last three actions before the current action
                        action_sequence = np.append(['null', 'null', 'null'], st_problem[:, ind(h, 'action_name')])
                        past_actions = np.array(
                            ['_'.join(action_sequence[j:(j + 3)].tolist()) for j in
                             range(len(st_problem))], dtype=str).reshape((-1, 1))

                        # the last two actions plus the current action
                        past_actions_with_current = np.array([
                            '_'.join(action_sequence[(j + 1):(j + 4)].tolist()) for j in
                            range(len(st_problem))], dtype=str).reshape((-1, 1))

                        # wheel spinning on the next assignment
                        next_assignment_ws = np.array(np.ones((len(st_problem))) * float(next_ws[np.argwhere(
                            np.array(next_ws[:, 0]) == st_problem[0, ind(h, 'assignment_id')]).ravel(), 1][0]),
                                                      dtype=str).reshape((-1, 1))

                        # stopout on the next assignment
                        next_assignment_so = np.array(np.ones((len(st_problem))) * float(next_so[np.argwhere(
                            np.array(next_so[:, 0]) == st_problem[0, ind(h, 'assignment_id')]).ravel(), 1][0]),
                                                      dtype=str).reshape((-1, 1))

                        # append the new columns
                        st_problem = np.append(st_problem, penultimate_hint, axis=1)
                        st_problem = np.append(st_problem, bottom_hint, axis=1)
                        st_problem = np.append(st_problem, past_actions, axis=1)
                        st_problem = np.append(st_problem, past_actions_with_current, axis=1)
                        st_problem = np.append(st_problem, next_assignment_ws, axis=1)
                        st_problem = np.append(st_problem, next_assignment_so, axis=1)

                        # write the student data to file (append after creating the file)
                        success = False
                        start = time.time()
                        while not success:
                            try:
                                if not written:
                                    du.write_csv(st_problem, outfile, header_out)
                                    written = True
                                else:
                                    du.write_csv(st_problem, outfile, append=True)
                                success = True
                            except PermissionError:
                                if time.time()-start < write_timout:
                                    print('\nError trying to write to file. Retrying...')
                                    old_str = ''
                                    time.sleep(1)
                                else:
                                    print('\nERROR: TIMEOUT | Unable to write to file')
                                    exit(0)


                # move on to the next student's data
                student = line.reshape((-1, len(h)))

            # print progress
            if not round((i / n_lines) * 100, 2) == round(((i - 1) / n_lines) * 100, 2):
                sys.stdout.write('\r' + (' ' * len(old_str)))
                output_str = '\r-- loading {}...({}%)'.format(filename, round((i / n_lines) * 100, 2))
                sys.stdout.write(output_str)
                sys.stdout.flush()
                old_str = output_str

            # increase line counter
            i += 1

        # print progress when finished
        sys.stdout.write('\r' + (' ' * len(old_str)))
        sys.stdout.write('\r-- loading {}...({}%)\n'.format(filename, 100))
        sys.stdout.flush()


if __name__ == "__main__":
    start = time.time()
    generate_features('resources/skill_builders_16_17.csv')
    print('Finished in {:<.1f} seconds'.format(time.time()-start))
