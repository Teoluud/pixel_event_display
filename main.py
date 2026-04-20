import sys

from data_processing import DataProcessing


if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser(usage= 'usage: %prog [options] run_id event_id')
    parser.add_option('-p', '--path', type=str, dest='p', default='../events/mcElectrons/csv', help='path to the import data')
    parser.add_option('-v', '--view', type=str, dest='v', default='x', help='select the view of the projection')

    (opts, args) = parser.parse_args()
    
    if len(args) == 0:
        sys.exit('Please provide run and event id.')

    path = opts.p
    run_id = args[0]
    event_id = args[1]

    pipeline = DataProcessing(path, run_id, event_id)
    pipeline.process_and_display(view=opts.v)