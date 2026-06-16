#lang racket
(define (strange_sort_list lst)
  (let loop ([s (sort lst <)] [take-min #t] [acc '()])
    (cond [(null? s) (reverse acc)]
          [take-min (loop (cdr s) #f (cons (car s) acc))]
          [else (loop (drop-right s 1) #t (cons (last s) acc))])))
