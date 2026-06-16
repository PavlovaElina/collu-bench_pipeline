#lang racket
(define (get_row lst x)
  (define coords
    (append-map
     (lambda (row)
       (define r (list-ref lst row))
       (for/list ([col (in-range (length r))] #:when (= (list-ref r col) x))
         (list row col)))
     (range (length lst))))
  (sort coords (lambda (a b)
                 (if (= (car a) (car b)) (> (cadr a) (cadr b)) (< (car a) (car b))))))
