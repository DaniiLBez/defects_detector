import argparse
from pathlib import Path
from defects_detector.preprocessing.plane_remover.service import PreprocessingService
from defects_detector.preprocessing.plane_remover.mvtec import MVTec3DPreprocessingService

def main():
    parser = argparse.ArgumentParser(description='Preprocess point clouds')

    # Create mutually exclusive group for processing options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dataset_path', type=str,
                      help='The root path of the MVTec 3D-AD dataset (for processing all files)')
    group.add_argument('--file_path', type=str,
                      help='Path to a specific tiff file to preprocess')

    parser.add_argument('--quiet', action='store_true',
                       help='Suppress output messages')

    args = parser.parse_args()

    # Create the MVTec3D preprocessor
    mvtec_preprocessor = MVTec3DPreprocessingService()

    # Create the preprocessing service with the specific implementation
    service = PreprocessingService(mvtec_preprocessor)

    if args.file_path:
        # Process single file
        file_path = Path(args.file_path)
        if not file_path.exists() or file_path.suffix != '.tiff':
            print(f"Error: File '{file_path}' doesn't exist or is not a TIFF file.")
            return

        if not args.quiet:
            print(f"Processing file: {file_path}")
        service.preprocess_file(file_path)
        if not args.quiet:
            print("Processing complete!")
    else:
        # Process all files in dataset
        processed_count = service.preprocess_all(args.dataset_path, verbose=not args.quiet)
        if not args.quiet:
            print(f"Preprocessing complete! Processed {processed_count} files.")

if __name__ == "__main__":
    main()