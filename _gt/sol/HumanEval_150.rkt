#lang racket
(define (x_or_y n x y)
  (define (prime? k) (and (> k 1) (for/and ([d (in-range 2 (+ 1 (integer-sqrt k)))]) (not (= 0 (modulo k d))))))
  (if (prime? n) x y))
