#lang racket
(define (count_up_to n)
  (define (prime? x)
    (and (> x 1) (for/and ([d (in-range 2 (+ 1 (integer-sqrt x)))]) (not (= 0 (modulo x d))))))
  (for/list ([x (in-range 2 n)] #:when (prime? x)) x))
