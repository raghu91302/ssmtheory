"""Reproduce Section 6: distance preservation under merge."""
from sheet_code_surgery import verify_merge_distance


def main():
    for L in [4, 6]:
        print(f"\nL = {L}:")
        result = verify_merge_distance(L)
        for k, v in result.items():
            print(f"  {k}: {v}")
        if result['ok']:
            print(f"  ok Distance preserved at d = {L}")
        else:
            print(f"  fail Distance check failed")


if __name__ == '__main__':
    main()
