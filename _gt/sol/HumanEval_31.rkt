#lang racket
(define (is_prime n)
  (cond [(< n 2) #f]
        [else (for/and ([d (in-range 2 (+ 1 (integer-sqrt n)))])
                (not (= 0 (modulo n d))))]))
