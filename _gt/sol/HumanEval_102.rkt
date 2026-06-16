#lang racket
(define (choose_num x y)
  (cond [(> x y) -1]
        [(even? y) y]
        [(>= (- y 1) x) (- y 1)]
        [else -1]))
