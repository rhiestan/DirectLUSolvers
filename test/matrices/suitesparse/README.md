# SuiteSparse test matrices

The archives in this directory are matrices from the
[SuiteSparse Matrix Collection](https://sparse.tamu.edu) (formerly the University of Florida
Sparse Matrix Collection), used by the DirectLUSolvers test suites. They are listed, with the
reason for choosing each, in [`../suitesparse.manifest`](../suitesparse.manifest).

## License and attribution

The matrices are licensed under the
[Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/)
(CC BY 4.0); the full text is in [`../CC-BY_4.0_License.txt`](../CC-BY_4.0_License.txt). They
are not covered by the license of the DirectLUSolvers source code.

Source: the SuiteSparse Matrix Collection, https://sparse.tamu.edu. Each archive
`<Group>/<Name>.tar.gz` comes from https://sparse.tamu.edu/MM/<Group>/<Name>.tar.gz, and its
original authors and history are on the matrix's page, https://sparse.tamu.edu/<Group>/<Name>.

Please cite the collection as:

> Timothy A. Davis and Yifan Hu. 2011. The University of Florida Sparse Matrix Collection.
> *ACM Transactions on Mathematical Software* 38, 1, Article 1 (December 2011), 25 pages.
> https://doi.org/10.1145/2049662.2049663

**Changes made:** each archive was repackaged to hold only the matrix itself,
`<Name>/<Name>.mtx`, byte-for-byte as distributed. The auxiliary files of the original archives
(right-hand sides, solutions, node coordinates and the like) were left out, since the tests do
not use them.

## Use

CMake unpacks the archives into the git-ignored `../cache/` at configure time
(`DLU_UNPACK_SUITESPARSE`, on by default). To adopt another matrix, see
[`../fetch_suitesparse.py`](../fetch_suitesparse.py).
