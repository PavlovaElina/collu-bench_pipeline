#lang racket
(define (prime_length s)
  (define n (string-length s))
  (and (> n 1) (for/and ([d (in-range 2 (+ 1 (integer-sqrt n)))]) (not (= 0 (modulo n d))))))
