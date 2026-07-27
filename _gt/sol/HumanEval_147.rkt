#lang racket
(define (get_max_triples n)
  (define a (for/vector ([i (in-range 1 (+ n 1))]) (+ (- (* i i) i) 1)))
  (for*/sum ([i (in-range n)] [j (in-range (+ i 1) n)] [k (in-range (+ j 1) n)]
             #:when (= 0 (modulo (+ (vector-ref a i) (vector-ref a j) (vector-ref a k)) 3)))
    1))
