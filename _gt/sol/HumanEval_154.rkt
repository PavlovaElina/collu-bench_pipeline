#lang racket
(define (cycpattern_check a b)
  (define n (string-length b))
  (for/or ([k (in-range n)])
    (string-contains? a (string-append (substring b k n) (substring b 0 k)))))
