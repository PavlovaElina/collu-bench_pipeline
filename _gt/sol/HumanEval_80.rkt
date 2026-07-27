#lang racket
(define (is_happy s)
  (define cs (list->vector (string->list s)))
  (define n (vector-length cs))
  (and (>= n 3)
       (for/and ([i (in-range (- n 2))])
         (not (or (char=? (vector-ref cs i) (vector-ref cs (+ i 1)))
                  (char=? (vector-ref cs (+ i 1)) (vector-ref cs (+ i 2)))
                  (char=? (vector-ref cs i) (vector-ref cs (+ i 2))))))))
