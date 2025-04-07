import argparse

from defects_detector.preprocessing.patches.mvtec import MVTec3DLoader, MVTec3DPatchCutter
from defects_detector.preprocessing.patches.patch_cutter import PatchCutterService


def main():
    parser = argparse.ArgumentParser(description='Cut patches from MVTec3D-AD dataset')
    parser.add_argument('--datasets_path', type=str, required=True, help='Path to MVTec3D-AD dataset')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save patches')
    parser.add_argument('--point_num', type=int, default=500, help='Number of points per patch')
    parser.add_argument('--group_mul', type=float, default=10.0, help='Group size multiplier')
    parser.add_argument('--split', type=str, choices=['train', 'test', 'validation', 'pretrain', 'all'],
                        default='test', help='Dataset split to process')

    args = parser.parse_args()

    loader = MVTec3DLoader()
    patch_cutter = MVTec3DPatchCutter(group_size=args.point_num, group_mul=args.group_mul)

    # Create service
    service = PatchCutterService(
        data_loader=loader,
        patch_cutter=patch_cutter,
        save_path=args.save_path
    )

    # Process files
    if args.split == 'all':
        splits = ['train', 'test', 'validation']
    else:
        splits = [args.split]

    for split in splits:
        service.process_directory(args.datasets_path, split)

    print(f"All patches saved to {args.save_path}")


if __name__ == "__main__":
    main()