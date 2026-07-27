#lang racket
(define (anti_shuffle s)
  (string-join
   (map (lambda (w) (list->string (sort (string->list w) char<?)))
        (string-split s " " #:trim? #f))
   " "))
