import sys

from data_processing import DataProcessing, BatchProcessor


if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser(usage= 'usage: %prog [options] [run_id event_id]') # Updated usage string
    parser.add_option('-p', '--path', type=str, dest='p', default='../events/mcElectrons/csv', help='path to the import data')
    parser.add_option('-v', '--view', type=str, dest='v', default='x', help='select the view of the projection')
    parser.add_option('-b', '--batch', dest='batch', default=False, action='store_true', help='process a piped input and save compressed numpy')

    (opts, args) = parser.parse_args()
    
    if opts.batch:
        # In batch mode, we don't need positional args; 
        # the stream provides the IDs.
        batch_processor = BatchProcessor()
        batch_processor.process_and_save(view=opts.v)
    else:
        # In standard mode, we MUST have run_id and event_id
        if len(args) < 2:
            sys.exit('Error: Please provide run_id and event_id for single event display.')
        path = opts.p
        run_id = args[0]
        event_id = args[1]
        pipeline = DataProcessing(path, run_id, event_id)
        pipeline.process_and_display(view=opts.v)