# hacked together code to read, edit, and rewrite a .grd file.
# 'clean' refers to the error where 1.242E-110 is written to file as
# 1.242-110.
# code searches for these, and fixes these

import optparse
import sys
import pdb


# optparse it!
usage = "usage: %prog  <input_file>"
parser = optparse.OptionParser(usage)
#parser.add_option('--t1', dest='title1', action='store', type='str', default='Open Dragone', 
#                  help='Title for first set of data.')

## file to read in is first arg.
(option, args) = parser.parse_args()

#print args
#print args[0][:-4]+'_clean.grd' ,'w+'

#sys.exit()

## open file to write to
with open(args[0][:-4]+'_clean.grd' ,'w+') as outfile:
    ## open file to read from
    header = True
    with open(args[0],'r') as infile:
        for line in infile:
            # read past header. need to skip 5 more lines! grrr...
            if line == '++++\n':
                count_now = True # start a line counter.
                count = 0
                header = False
            if header:  # just write same line while in header.
                outfile.write(line)
            elif count < 6 and count_now:  #  write same lines in info section 
                outfile.write(line)
                count += 1 # iterate my line counter.
            else:   # now read/fix/write the data.
                n_vals = ['','','','']
                for i,val in enumerate(line.split()):
                    if not('E' in val):                   # test if error exists
                       n_vals[i] = val[:-4]+'E'+val[-4:]     #fix error
                    else:
                       n_vals[i] = val
                #write line (or fixed line) to outfile.
                outfile.write('  '.join(n_vals)+'\n')

print 'File fixed!! We hope.\n   Written to %s' %(args[0][:-4]+'_clean.grd' )
