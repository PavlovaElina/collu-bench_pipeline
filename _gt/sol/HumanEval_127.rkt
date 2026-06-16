#lang racket
(define (intersection interval1 interval2)
  (define (prime? x) (and (> x 1) (for/and ([d (in-range 2 (+ 1 (integer-sqrt x)))]) (not (= 0 (modulo x d))))))
  (define lo (max (first interval1) (first interval2)))
  (define hi (min (second interval1) (second interval2)))
  (define len (- hi lo))
  (if (and (> len 0) (prime? len)) "YES" "NO"))
